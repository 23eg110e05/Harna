import re

# Category keyword rules for supermarket aisle mapping
AISLE_RULES = {
    'produce': {
        'title': 'Fresh Produce',
        'icon': 'fa-apple-alt',
        'keywords': [
            'onion', 'tomato', 'potato', 'garlic', 'ginger', 'chilli', 'chili', 'coriander', 
            'cilantro', 'lemon', 'lime', 'curry leaves', 'mint', 'spinach', 'karela', 'carrot', 
            'cabbage', 'cauliflower', 'capsicum', 'bell pepper', 'cucumber', 'peas', 'beans', 
            'methi', 'fenugreek leaves', 'brinjal', 'eggplant', 'aubergine', 'mushroom', 
            'palak', 'gourd', 'bhindi', 'okra', 'ladies finger', 'radish', 'beetroot', 
            'pumpkin', 'drumstick', 'spring onion', 'apple', 'banana', 'mango', 'coconut',
            'avocado', 'zucchini', 'broccoli', 'corn', 'lettuce', 'celery'
        ]
    },
    'dairy_eggs': {
        'title': 'Dairy & Eggs',
        'icon': 'fa-egg',
        'keywords': [
            'milk', 'curd', 'yogurt', 'yoghurt', 'paneer', 'cheese', 'butter', 'ghee', 
            'cream', 'malai', 'egg', 'eggs', 'buttermilk', 'whey', 'khoya', 'condensed milk'
        ]
    },
    'meat_seafood': {
        'title': 'Meat & Seafood',
        'icon': 'fa-drumstick-bite',
        'keywords': [
            'chicken', 'mutton', 'fish', 'prawn', 'prawns', 'shrimp', 'lamb', 'beef', 
            'pork', 'turkey', 'keema', 'salmon', 'tuna', 'crab'
        ]
    },
    'grains_flours': {
        'title': 'Grains, Flours & Lentils',
        'icon': 'fa-seedling',
        'keywords': [
            'rice', 'flour', 'atta', 'maida', 'besan', 'sooji', 'rava', 'semolina', 
            'dal', 'dhal', 'lentil', 'lentils', 'chana', 'rajma', 'moong', 'urad', 
            'toor', 'masoor', 'poha', 'oats', 'noodle', 'noodles', 'pasta', 'bread', 
            'quinoa', 'millet', 'ragi', 'vermicelli', 'semiya', 'sabudana', 'sago'
        ]
    },
    'spices_seasonings': {
        'title': 'Spices & Seasonings',
        'icon': 'fa-mortar-pestle',
        'keywords': [
            'turmeric', 'haldi', 'cumin', 'jeera', 'mustard', 'rai', 'coriander powder', 
            'dhania', 'red chilli powder', 'garam masala', 'cardamom', 'clove', 'cinnamon', 
            'bay leaf', 'peppercorn', 'black pepper', 'fennel', 'saunf', 'asafoetida', 
            'hing', 'amchur', 'dry mango', 'chaat masala', 'kasuri methi', 'nutmeg', 
            'saffron', 'star anise', 'sesame', 'til', 'salt', 'black salt'
        ]
    },
    'oils_sauces': {
        'title': 'Oils & Sauces',
        'icon': 'fa-wine-bottle',
        'keywords': [
            'oil', 'sunflower oil', 'mustard oil', 'sesame oil', 'olive oil', 'coconut oil', 
            'vegetable oil', 'vinegar', 'soy sauce', 'chilli sauce', 'tomato sauce', 
            'ketchup', 'honey', 'sugar', 'jaggery', 'tamarind', 'imli'
        ]
    },
    'nuts_pantry': {
        'title': 'Nuts, Seeds & Pantry',
        'icon': 'fa-box',
        'keywords': [
            'cashew', 'almond', 'badam', 'kaju', 'raisin', 'kishmish', 'walnut', 
            'pistachio', 'peanut', 'groundnut', 'baking powder', 'baking soda', 
            'yeast', 'vanilla', 'cocoa', 'chocolate', 'water'
        ]
    }
}

def clean_ingredient_line(raw):
    """Normalize and clean ingredient strings."""
    if not raw or not isinstance(raw, str):
        return ""
    # Strip unnecessary trailing instructions like '- finely chopped'
    item = raw.strip()
    return item

def classify_ingredient(ingredient_text):
    """Map an ingredient name to an aisle category."""
    lower = ingredient_text.lower()
    for cat_id, meta in AISLE_RULES.items():
        for kw in meta['keywords']:
            # word boundary search for precision
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                return cat_id
    return 'other'

def generate_aisle_grocery_list(recipe_ingredients_list):
    """
    Given a list of ingredient strings or comma-delimited strings from planned recipes,
    group them into supermarket aisles.
    """
    aisles = {
        'produce': {'title': AISLE_RULES['produce']['title'], 'icon': AISLE_RULES['produce']['icon'], 'ingredients': []},
        'dairy_eggs': {'title': AISLE_RULES['dairy_eggs']['title'], 'icon': AISLE_RULES['dairy_eggs']['icon'], 'ingredients': []},
        'meat_seafood': {'title': AISLE_RULES['meat_seafood']['title'], 'icon': AISLE_RULES['meat_seafood']['icon'], 'ingredients': []},
        'grains_flours': {'title': AISLE_RULES['grains_flours']['title'], 'icon': AISLE_RULES['grains_flours']['icon'], 'ingredients': []},
        'spices_seasonings': {'title': AISLE_RULES['spices_seasonings']['title'], 'icon': AISLE_RULES['spices_seasonings']['icon'], 'ingredients': []},
        'oils_sauces': {'title': AISLE_RULES['oils_sauces']['title'], 'icon': AISLE_RULES['oils_sauces']['icon'], 'ingredients': []},
        'nuts_pantry': {'title': AISLE_RULES['nuts_pantry']['title'], 'icon': AISLE_RULES['nuts_pantry']['icon'], 'ingredients': []},
        'other': {'title': 'Other Pantry Items', 'icon': 'fa-shopping-basket', 'ingredients': []}
    }

    seen_items = set()

    for ing_blob in recipe_ingredients_list:
        if not ing_blob:
            continue
        parts = [p.strip() for p in ing_blob.split(',') if p.strip()]
        for p in parts:
            cleaned = clean_ingredient_line(p)
            if not cleaned or len(cleaned) < 2:
                continue
            normalized = cleaned.lower()
            if normalized in seen_items:
                continue
            seen_items.add(normalized)
            cat = classify_ingredient(cleaned)
            aisles[cat]['ingredients'].append(cleaned)

    # Filter out empty aisles
    return {k: v for k, v in aisles.items() if v['ingredients']}
