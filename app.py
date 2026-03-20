from flask import Flask, render_template, request, jsonify, Blueprint, redirect, url_for
from flask_login import LoginManager, login_required, current_user, login_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from models import db, User, Basket, BasketItem
from auth import auth
import pandas as pd
from scipy.optimize import linprog
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
login_manager.login_view = 'auth.login'   # არ გადავამისამართოთ — სტუმარი დაშვებულია

limiter = Limiter(key_func=get_remote_address, app=app,
                  default_limits=[], storage_uri='memory://')

app.register_blueprint(auth)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# ── Products ──────────────────────────────────────────────────────────────────
def load_products(excluded_names=None):
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
    if excluded_names:
        df = df[~df['product'].isin(excluded_names)]
    return df.reset_index(drop=True)

# ── Optimizer ─────────────────────────────────────────────────────────────────
def run_optimizer(target_calories, target_protein, target_carbs,
                  target_fat, max_packages, excluded_names=None,
                  previous_items=None):

    use_macro_mode = any(x is not None for x in [target_protein, target_carbs, target_fat])
    if not use_macro_mode:
        target_protein = (target_calories * 0.40) / 4
        target_carbs   = (target_calories * 0.30) / 4
        target_fat     = (target_calories * 0.30) / 9
        calorie_mode   = 'soft'
    else:
        target_protein = float(target_protein or 0)
        target_carbs   = float(target_carbs   or 0)
        target_fat     = float(target_fat     or 0)
        calorie_mode   = 'cap'

    prev_cal = prev_prot = prev_carb = prev_fat = 0.0
    if previous_items:
        for pi in previous_items:
            f = pi['remaining_grams'] / 100.0
            prev_cal  += pi.get('cal_per_100',  0) * f
            prev_prot += pi.get('prot_per_100', 0) * f
            prev_carb += pi.get('carb_per_100', 0) * f
            prev_fat  += pi.get('fat_per_100',  0) * f

    eff_cal  = max(0, target_calories - prev_cal)
    eff_prot = max(0, target_protein  - prev_prot)
    eff_carb = max(0, target_carbs    - prev_carb)
    eff_fat  = max(0, target_fat      - prev_fat)

    df = load_products(excluded_names)
    n  = len(df)
    c  = df['price_gel'].values.astype(float)

    A_ub, b_ub = [], []
    if calorie_mode == 'cap':
        A_ub.append(-df['kcal_per_package'].values.astype(float)); b_ub.append(-eff_cal * 0.95)
        A_ub.append( df['kcal_per_package'].values.astype(float)); b_ub.append( eff_cal)
    else:
        A_ub.append(-df['kcal_per_package'].values.astype(float)); b_ub.append(-eff_cal * 0.95)
        A_ub.append( df['kcal_per_package'].values.astype(float)); b_ub.append( eff_cal * 1.05)

    A_ub.append(-df['protein_per_package'].values.astype(float)); b_ub.append(-eff_prot)
    A_ub.append(-df['carbs_per_package'].values.astype(float));   b_ub.append(-eff_carb)
    A_ub.append(-df['fats_per_package'].values.astype(float));    b_ub.append(-eff_fat)

    bounds = [(0, float(max_packages)) for _ in range(n)]
    result = linprog(np.array(c), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=bounds, method='highs')

    if result.status != 0:
        return None, 'ოპტიმიზაცია ვერ მოხერხდა. სცადე კალორიების გაზრდა ან მაქს. შეფუთვების გაზრდა.'

    basket = []
    total_cost = total_cal = total_prot = total_carb = total_fat_v = 0

    for i, row in df.iterrows():
        units = result.x[i]
        if units < 0.01: continue
        units_r = round(units * 2) / 2
        if units_r < 0.5: continue

        pkg_size  = row['package_size']
        rec_grams = min(units_r * pkg_size, 200 * units_r)
        rem_grams = max(0, units_r * pkg_size - rec_grams)

        if   row['piece'] != 0: pkg_label = f"{int(row['piece'])} ც."
        elif row['kg']    != 0: pkg_label = f"{row['kg']} კგ"
        elif row['l']     != 0: pkg_label = f"{row['l']} ლ"
        else:                   pkg_label = f"{int(row['ml'])} მლ"

        factor = rec_grams / 100.0
        cost = round(units_r * row['price_gel'], 2)
        cal  = round(row['kcal']    * factor, 1)
        prot = round(row['protein'] * factor, 1)
        carb = round(row['carbs']   * factor, 1)
        fat  = round(row['fats']    * factor, 1)

        basket.append({
            'name': row['product'], 'units': units_r, 'pkg_label': pkg_label,
            'recommended_grams': round(rec_grams, 1), 'remaining_grams': round(rem_grams, 1),
            'cost': cost, 'calories': cal, 'protein': prot, 'carbs': carb, 'fat': fat,
            'cal_per_100': row['kcal'], 'prot_per_100': row['protein'],
            'carb_per_100': row['carbs'], 'fat_per_100': row['fats'],
        })
        total_cost += cost; total_cal += cal; total_prot += prot
        total_carb += carb; total_fat_v += fat

    total_cal += prev_cal; total_prot += prev_prot
    total_carb += prev_carb; total_fat_v += prev_fat

    return {
        'basket': basket,
        'totals': {
            'cost': round(total_cost, 2), 'calories': round(total_cal, 1),
            'protein': round(total_prot, 1), 'carbs': round(total_carb, 1),
            'fat': round(total_fat_v, 1),
        },
        'targets': {
            'calories': target_calories, 'protein': round(target_protein, 1),
            'carbs': round(target_carbs, 1), 'fat': round(target_fat, 1),
        },
        'mode': calorie_mode,
    }, None

# ── Routes ────────────────────────────────────────────────────────────────────
main = Blueprint('main', __name__)

@main.route('/')
def index():
    # ყველა პირდაპირ კალკულატორზე — logged in თუ არა
    last_basket = None
    if current_user.is_authenticated:
        last_basket = (Basket.query
                       .filter_by(user_id=current_user.id)
                       .order_by(Basket.created_at.desc())
                       .first())
    return render_template('app.html', last_basket=last_basket)

# /app alias (backward compat)
@main.route('/app')
def app_page():
    return redirect(url_for('main.index'))

@main.route('/optimize', methods=['POST'])
@limiter.limit('15 per minute')
def optimize():
    data = request.get_json()
    excluded       = data.get('excluded', [])
    use_previous   = data.get('use_previous', False)
    prev_basket_id = data.get('prev_basket_id')
    previous_items = []

    # წინა კალათა მხოლოდ logged-in მომხმარებლისთვის
    if use_previous and prev_basket_id and current_user.is_authenticated:
        prev = Basket.query.filter_by(id=prev_basket_id, user_id=current_user.id).first()
        if prev:
            for item in prev.items:
                denom = (item.recommended_grams / 100) if item.recommended_grams else 1
                previous_items.append({
                    'product_name':    item.product_name,
                    'remaining_grams': item.remaining_grams,
                    'cal_per_100':     item.calories / denom,
                    'prot_per_100':    item.protein  / denom,
                    'carb_per_100':    item.carbs    / denom,
                    'fat_per_100':     item.fat      / denom,
                })

    result, error = run_optimizer(
        target_calories = float(data.get('calories', 2000)),
        target_protein  = data.get('protein'),
        target_carbs    = data.get('carbs'),
        target_fat      = data.get('fat'),
        max_packages    = int(data.get('max_packages', 3)),
        excluded_names  = excluded,
        previous_items  = previous_items,
    )
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)

@main.route('/basket/save', methods=['POST'])
@login_required   # შენახვა მხოლოდ logged-in
def save_basket():
    data        = request.get_json()
    basket_data = data.get('basket_data')
    name        = data.get('name', '').strip() or None

    b = Basket(
        user_id=current_user.id, name=name,
        target_calories=basket_data['targets']['calories'],
        target_protein=basket_data['targets'].get('protein'),
        target_carbs=basket_data['targets'].get('carbs'),
        target_fat=basket_data['targets'].get('fat'),
        mode=basket_data.get('mode','soft'),
        total_cost=basket_data['totals']['cost'],
        total_calories=basket_data['totals']['calories'],
        total_protein=basket_data['totals']['protein'],
        total_carbs=basket_data['totals']['carbs'],
        total_fat=basket_data['totals']['fat'],
    )
    db.session.add(b); db.session.flush()

    for item in basket_data['basket']:
        db.session.add(BasketItem(
            basket_id=b.id, product_name=item['name'], pkg_label=item['pkg_label'],
            units=item['units'], recommended_grams=item['recommended_grams'],
            remaining_grams=item['remaining_grams'], cost=item['cost'],
            calories=item['calories'], protein=item['protein'],
            carbs=item['carbs'], fat=item['fat'],
        ))
    db.session.commit()
    return jsonify({'id': b.id, 'message': 'კალათა შენახულია'})

@main.route('/history')
@login_required
def history():
    baskets = (Basket.query.filter_by(user_id=current_user.id)
               .order_by(Basket.created_at.desc()).all())
    return render_template('history.html', baskets=baskets)

@main.route('/basket/<int:basket_id>')
@login_required
def basket_detail(basket_id):
    basket = Basket.query.filter_by(id=basket_id, user_id=current_user.id).first_or_404()
    return render_template('basket_detail.html', basket=basket)

@main.route('/basket/<int:basket_id>/item/<int:item_id>/update', methods=['POST'])
@login_required
def update_item(basket_id, item_id):
    basket = Basket.query.filter_by(id=basket_id, user_id=current_user.id).first_or_404()
    item   = BasketItem.query.filter_by(id=item_id, basket_id=basket.id).first_or_404()
    new_consumed     = float(request.get_json().get('consumed_grams', item.recommended_grams))
    total            = item.recommended_grams + item.remaining_grams
    item.recommended_grams = min(new_consumed, total)
    item.remaining_grams   = max(0, total - new_consumed)
    db.session.commit()
    return jsonify({'ok': True, 'remaining': item.remaining_grams})

@main.route('/basket/<int:basket_id>/item/<int:item_id>/delete', methods=['POST'])
@login_required
def delete_item(basket_id, item_id):
    basket = Basket.query.filter_by(id=basket_id, user_id=current_user.id).first_or_404()
    item   = BasketItem.query.filter_by(id=item_id, basket_id=basket.id).first_or_404()
    db.session.delete(item); db.session.commit()
    return jsonify({'ok': True})

@main.route('/basket/<int:basket_id>/delete', methods=['POST'])
@login_required
def delete_basket(basket_id):
    basket = Basket.query.filter_by(id=basket_id, user_id=current_user.id).first_or_404()
    db.session.delete(basket); db.session.commit()
    return jsonify({'ok': True})

app.register_blueprint(main)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
