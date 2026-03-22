from flask import Flask, render_template, request, jsonify, Blueprint, redirect, url_for
from flask_login import LoginManager, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from models import db, User, Basket, BasketItem
from auth import auth
import pandas as pd
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

app.config['SECRET_KEY']                     = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config['SQLALCHEMY_DATABASE_URI']        = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db_url = app.config['SQLALCHEMY_DATABASE_URI']
if db_url.startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url.replace('postgres://', 'postgresql://', 1)

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'
limiter = Limiter(key_func=get_remote_address, app=app,
                  default_limits=[], storage_uri='memory://')
app.register_blueprint(auth)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# ── კონფიგურაცია ──────────────────────────────────────────────────────────────
DIET_STYLES = {
    'balanced':    {'p': 0.25, 'c': 0.45, 'f': 0.30, 'label': '⚖️ დაბალანსებული'},
    'high_protein':{'p': 0.40, 'c': 0.30, 'f': 0.30, 'label': '💪 მაღალი ცილა'},
    'low_carb':    {'p': 0.30, 'c': 0.10, 'f': 0.60, 'label': '🥩 დაბალი ნახშ.'},
    'custom':      {'p': None, 'c': None, 'f': None,  'label': '🎯 მაკრო'},
}

CONDIMENT_KEYWORDS = ['ზეთი', 'oil', 'salt', 'მარილი', 'მდოგვი',
                      'sauce', 'სოუს', 'ძმარი', 'vinegar', 'საფუარი']
CONDIMENT_MAX_GRAMS = 15
MAX_GRAMS_PER_ITEM  = 200

# ── CSV კეში ──────────────────────────────────────────────────────────────────
_products_df = None

def get_products():
    global _products_df
    if _products_df is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'products.csv')
        df = pd.read_csv(csv_path)

        def package_grams(row):
            if row['piece'] != 0: return row['piece']
            if row['kg']    != 0: return row['kg'] * 1000
            if row['l']     != 0: return row['l']  * 1000
            if row['ml']    != 0: return row['ml']
            return None

        df['package_size'] = df.apply(package_grams, axis=1)
        df = df[df['package_size'].notna() & (df['package_size'] > 0)]
        df = df[df['kcal_per_package'] > 0]

        # სანელებლების მონიშვნა
        def is_condiment(name):
            n = str(name).lower()
            return any(kw in n for kw in CONDIMENT_KEYWORDS)
        df['is_condiment'] = df['product'].apply(is_condiment)

        _products_df = df.reset_index(drop=True)
    return _products_df

def load_products(excluded_names=None):
    df = get_products()
    if excluded_names:
        df = df[~df['product'].isin(excluded_names)].reset_index(drop=True)
    return df

# ── მაკრო მიზნების გამოთვლა ──────────────────────────────────────────────────
def compute_targets(calories, style, protein=None, carbs=None, fat=None):
    if style == 'custom' and all(x is not None for x in [protein, carbs, fat]):
        return float(protein), float(carbs), float(fat), 'cap'

    ratios = DIET_STYLES.get(style, DIET_STYLES['balanced'])
    p = (calories * ratios['p']) / 4
    c = (calories * ratios['c']) / 4
    f = (calories * ratios['f']) / 9

    # სანიტარული შემოწმება
    macro_kcal = p * 4 + c * 4 + f * 9
    if macro_kcal > calories * 1.02:
        factor = calories / macro_kcal
        p, c, f = p * factor, c * factor, f * factor

    return round(p), round(c), round(f), 'soft'

# ── ოპტიმიზატორი ─────────────────────────────────────────────────────────────
def run_optimizer(target_calories, style, max_packages, max_ingredients,
                  excluded_names=None, previous_items=None,
                  protein=None, carbs=None, fat=None):

    target_protein, target_carbs, target_fat, calorie_mode = compute_targets(
        target_calories, style, protein, carbs, fat)

    # წინა კალათის მაკროები
    prev_cal = prev_prot = prev_carb = prev_fat = 0.0
    prev_product_names = []

    if previous_items:
        for pi in previous_items:
            if pi.get('remaining_grams', 0) <= 0:
                continue
            f = pi['remaining_grams'] / 100.0
            prev_cal  += pi.get('cal_per_100',  0) * f
            prev_prot += pi.get('prot_per_100', 0) * f
            prev_carb += pi.get('carb_per_100', 0) * f
            prev_fat  += pi.get('fat_per_100',  0) * f
            prev_product_names.append(pi['product_name'])

    eff_cal  = max(0, target_calories - prev_cal)
    eff_prot = max(0, target_protein  - prev_prot)
    eff_carb = max(0, target_carbs    - prev_carb)
    eff_fat  = max(0, target_fat      - prev_fat)

    all_excluded = list(excluded_names or []) + prev_product_names
    df = load_products(all_excluded)
    n  = len(df)

    c_obj = df['price_gel'].values.astype(float)

    # სანელებლების გრამის შეზღუდვა
    gram_caps = np.where(
        df['is_condiment'],
        CONDIMENT_MAX_GRAMS,
        MAX_GRAMS_PER_ITEM
    ).astype(float)

    # pkg_size-ზე გაყოფა → max fraction of package to eat
    eat_fractions = gram_caps / df['package_size'].values

    # შეზღუდვები: [კალ, ცილა, ნახშ, ცხიმი] × n
    A_rows, b_lo, b_hi = [], [], []

    if calorie_mode == 'cap':
        A_rows.append(df['kcal_per_package'].values.astype(float))
        b_lo.append(eff_cal * 0.95); b_hi.append(eff_cal)
    else:
        A_rows.append(df['kcal_per_package'].values.astype(float))
        b_lo.append(eff_cal * 0.95); b_hi.append(eff_cal * 1.05)

    A_rows.append(df['protein_per_package'].values.astype(float))
    b_lo.append(eff_prot); b_hi.append(np.inf)
    A_rows.append(df['carbs_per_package'].values.astype(float))
    b_lo.append(eff_carb); b_hi.append(np.inf)
    A_rows.append(df['fats_per_package'].values.astype(float))
    b_lo.append(eff_fat); b_hi.append(np.inf)

    constraints = LinearConstraint(np.array(A_rows),
                                   lb=np.array(b_lo), ub=np.array(b_hi))
    bounds = Bounds(lb=np.zeros(n), ub=np.full(n, int(max_packages)))

    result = milp(c_obj, constraints=constraints,
                  integrality=np.ones(n), bounds=bounds)

    if not result.success:
        # fallback continuous
        A_ub, b_ub = [], []
        if calorie_mode == 'cap':
            A_ub.append(-df['kcal_per_package'].values); b_ub.append(-eff_cal * 0.95)
            A_ub.append( df['kcal_per_package'].values); b_ub.append( eff_cal)
        else:
            A_ub.append(-df['kcal_per_package'].values); b_ub.append(-eff_cal * 0.95)
            A_ub.append( df['kcal_per_package'].values); b_ub.append( eff_cal * 1.05)
        A_ub.append(-df['protein_per_package'].values); b_ub.append(-eff_prot)
        A_ub.append(-df['carbs_per_package'].values);   b_ub.append(-eff_carb)
        A_ub.append(-df['fats_per_package'].values);    b_ub.append(-eff_fat)
        r2 = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=[(0, int(max_packages))]*n, method='highs')
        if r2.status != 0:
            return None, 'ოპტიმიზაცია ვერ მოხერხდა. სცადე კალორიების გაზრდა.'
        x = np.clip(np.ceil(r2.x).astype(int), 0, int(max_packages))
    else:
        x = np.clip(np.round(result.x).astype(int), 0, int(max_packages))

    # MAX_INGREDIENTS შეზღუდვა — თუ მეტი გამოვიდა, ყველაზე იაფი დავტოვოთ
    active = [(i, x[i]) for i in range(n) if x[i] > 0]
    if len(active) > max_ingredients:
        active.sort(key=lambda t: df.iloc[t[0]]['price_gel'] * t[1])
        keep = set(i for i, _ in active[:max_ingredients])
        x = np.array([x[i] if i in keep else 0 for i in range(n)])

    basket = []
    total_cost = total_cal = total_prot = total_carb = total_fat_v = 0

    # წინა კალათის პროდუქტები
    if previous_items:
        for pi in previous_items:
            if pi.get('remaining_grams', 0) <= 0:
                continue
            rem = pi['remaining_grams']
            fac = rem / 100.0
            cal  = round(pi.get('cal_per_100',  0) * fac, 1)
            prot = round(pi.get('prot_per_100', 0) * fac, 1)
            carb = round(pi.get('carb_per_100', 0) * fac, 1)
            fat  = round(pi.get('fat_per_100',  0) * fac, 1)
            basket.append({
                'name': pi['product_name'], 'units': 0,
                'pkg_label': 'წინა კალათა', 'is_previous': True,
                'recommended_grams': round(rem, 1), 'remaining_grams': 0,
                'cost': 0, 'calories': cal, 'protein': prot,
                'carbs': carb, 'fat': fat,
                'cal_per_100': pi.get('cal_per_100', 0),
                'prot_per_100': pi.get('prot_per_100', 0),
                'carb_per_100': pi.get('carb_per_100', 0),
                'fat_per_100': pi.get('fat_per_100', 0),
            })
            total_cal += cal; total_prot += prot
            total_carb += carb; total_fat_v += fat

    # ახალი პროდუქტები
    for i, row in df.iterrows():
        units = int(x[i])
        if units <= 0:
            continue

        pkg_size  = row['package_size']
        gram_cap  = CONDIMENT_MAX_GRAMS if row['is_condiment'] else MAX_GRAMS_PER_ITEM
        rec_grams = min(units * pkg_size, gram_cap * units)
        rem_grams = max(0, units * pkg_size - rec_grams)

        if   row['piece'] != 0: pkg_label = f"{int(row['piece'])} ც."
        elif row['kg']    != 0: pkg_label = f"{row['kg']} კგ"
        elif row['l']     != 0: pkg_label = f"{row['l']} ლ"
        else:                   pkg_label = f"{int(row['ml'])} მლ"

        cost = round(units * row['price_gel'], 2)
        fac  = rec_grams / 100.0
        cal  = round(row['kcal']    * fac, 1)
        prot = round(row['protein'] * fac, 1)
        carb = round(row['carbs']   * fac, 1)
        fat  = round(row['fats']    * fac, 1)

        basket.append({
            'name': row['product'], 'units': units,
            'pkg_label': pkg_label, 'is_previous': False,
            'recommended_grams': round(rec_grams, 1),
            'remaining_grams':   round(rem_grams, 1),
            'cost': cost, 'calories': cal, 'protein': prot,
            'carbs': carb, 'fat': fat,
            'cal_per_100': row['kcal'], 'prot_per_100': row['protein'],
            'carb_per_100': row['carbs'], 'fat_per_100': row['fats'],
        })
        total_cost += cost; total_cal += cal; total_prot += prot
        total_carb += carb; total_fat_v += fat

    return {
        'basket': basket,
        'totals': {
            'cost':     round(total_cost, 2),
            'calories': round(total_cal, 1),
            'protein':  round(total_prot, 1),
            'carbs':    round(total_carb, 1),
            'fat':      round(total_fat_v, 1),
        },
        'targets': {
            'calories': target_calories,
            'protein':  round(target_protein, 1),
            'carbs':    round(target_carbs, 1),
            'fat':      round(target_fat, 1),
        },
        'mode': calorie_mode,
        'style': style,
    }, None

# ── Routes ────────────────────────────────────────────────────────────────────
main = Blueprint('main', __name__)

@main.route('/')
def index():
    last_basket = None
    if current_user.is_authenticated:
        last_basket = (Basket.query.filter_by(user_id=current_user.id)
                       .order_by(Basket.created_at.desc()).first())
    return render_template('app.html',
                           last_basket=last_basket,
                           diet_styles=DIET_STYLES)

@main.route('/app')
def app_page():
    return redirect(url_for('main.index'))

@main.route('/optimize', methods=['POST'])
@limiter.limit('15 per minute')
def optimize():
    data           = request.get_json()
    excluded       = data.get('excluded', [])
    use_previous   = data.get('use_previous', False)
    prev_basket_id = data.get('prev_basket_id')
    previous_items = []

    if use_previous and prev_basket_id and current_user.is_authenticated:
        prev = Basket.query.filter_by(id=prev_basket_id,
                                      user_id=current_user.id).first()
        if prev:
            for item in prev.items:
                if item.remaining_grams <= 0:
                    continue
                denom = (item.recommended_grams / 100) if item.recommended_grams else 1
                previous_items.append({
                    'product_name':    item.product_name,
                    'remaining_grams': item.remaining_grams,
                    'cal_per_100':  item.calories / denom,
                    'prot_per_100': item.protein  / denom,
                    'carb_per_100': item.carbs    / denom,
                    'fat_per_100':  item.fat      / denom,
                })

    result, error = run_optimizer(
        target_calories  = float(data.get('calories', 2000)),
        style            = data.get('style', 'balanced'),
        max_packages     = int(data.get('max_packages', 3)),
        max_ingredients  = int(data.get('max_ingredients', 8)),
        excluded_names   = excluded,
        previous_items   = previous_items,
        protein          = data.get('protein'),
        carbs            = data.get('carbs'),
        fat              = data.get('fat'),
    )
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)

@main.route('/basket/save', methods=['POST'])
@login_required
def save_basket():
    data       = request.get_json()
    bd         = data.get('basket_data')
    replace_id = data.get('replace_basket_id')

    if replace_id:
        old = Basket.query.filter_by(id=replace_id,
                                     user_id=current_user.id).first()
        if old:
            db.session.delete(old)

    b = Basket(
        user_id=current_user.id,
        target_calories=bd['targets']['calories'],
        target_protein=bd['targets'].get('protein'),
        target_carbs=bd['targets'].get('carbs'),
        target_fat=bd['targets'].get('fat'),
        mode=bd.get('mode', 'soft'),
        total_cost=bd['totals']['cost'],
        total_calories=bd['totals']['calories'],
        total_protein=bd['totals']['protein'],
        total_carbs=bd['totals']['carbs'],
        total_fat=bd['totals']['fat'],
    )
    db.session.add(b); db.session.flush()
    for item in bd['basket']:
        db.session.add(BasketItem(
            basket_id=b.id, product_name=item['name'],
            pkg_label=item['pkg_label'], units=item['units'],
            recommended_grams=item['recommended_grams'],
            remaining_grams=item['remaining_grams'],
            cost=item['cost'], calories=item['calories'],
            protein=item['protein'], carbs=item['carbs'], fat=item['fat'],
        ))
    db.session.commit()
    return jsonify({'id': b.id})

@main.route('/history')
@login_required
def history():
    baskets = (Basket.query.filter_by(user_id=current_user.id)
               .order_by(Basket.created_at.desc()).all())
    return render_template('history.html', baskets=baskets)

@main.route('/basket/<int:basket_id>')
@login_required
def basket_detail(basket_id):
    basket = Basket.query.filter_by(id=basket_id,
                                    user_id=current_user.id).first_or_404()
    return render_template('basket_detail.html', basket=basket)

@main.route('/basket/<int:basket_id>/item/<int:item_id>/update', methods=['POST'])
@login_required
def update_item(basket_id, item_id):
    basket = Basket.query.filter_by(id=basket_id,
                                    user_id=current_user.id).first_or_404()
    item   = BasketItem.query.filter_by(id=item_id,
                                        basket_id=basket.id).first_or_404()
    new_c = float(request.get_json().get('consumed_grams', item.recommended_grams))
    total = item.recommended_grams + item.remaining_grams
    item.recommended_grams = min(new_c, total)
    item.remaining_grams   = max(0, total - new_c)
    db.session.commit()
    return jsonify({'ok': True, 'remaining': item.remaining_grams})

@main.route('/basket/<int:basket_id>/item/<int:item_id>/delete', methods=['POST'])
@login_required
def delete_item(basket_id, item_id):
    basket = Basket.query.filter_by(id=basket_id,
                                    user_id=current_user.id).first_or_404()
    item   = BasketItem.query.filter_by(id=item_id,
                                        basket_id=basket.id).first_or_404()
    db.session.delete(item); db.session.commit()
    return jsonify({'ok': True})

@main.route('/basket/<int:basket_id>/delete', methods=['POST'])
@login_required
def delete_basket(basket_id):
    basket = Basket.query.filter_by(id=basket_id,
                                    user_id=current_user.id).first_or_404()
    db.session.delete(basket); db.session.commit()
    return jsonify({'ok': True})

app.register_blueprint(main)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
