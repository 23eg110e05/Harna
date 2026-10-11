import os
import re
import ast
import json
import math
import logging
import urllib.request
import urllib.error
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# CULINARY FRACTION PARSING & FORMATTING
# ============================================================================

UNICODE_FRACTIONS = {
    '½': 0.5,
    '⅓': 1.0 / 3.0,
    '⅔': 2.0 / 3.0,
    '¼': 0.25,
    '¾': 0.75,
    '⅕': 0.2,
    '⅖': 0.4,
    '⅗': 0.6,
    '⅘': 0.8,
    '⅙': 1.0 / 6.0,
    '⅚': 5.0 / 6.0,
    '⅛': 0.125,
    '⅜': 0.375,
    '⅝': 0.625,
    '⅞': 0.875,
}

def parse_culinary_fraction(val_str):
    """
    Robustly parses culinary fractions, mixed numbers, hyphenated fractions ('1-1/2'),
    unicode fractions ('½', '1 ½'), and decimals ('1.5') to float.
    """
    if val_str is None:
        raise ValueError("Cannot parse None as culinary fraction")
    if isinstance(val_str, (int, float)):
        return float(val_str)

    s = str(val_str).strip()
    if not s:
        raise ValueError("Empty string provided")

    # 1. Unicode fraction characters (e.g. '½', '1 ½', '1½')
    for u_char, u_val in UNICODE_FRACTIONS.items():
        if u_char in s:
            parts = s.split(u_char)
            prefix = parts[0].strip()
            whole = float(prefix) if prefix else 0.0
            return whole + u_val

    # 2. Hyphenated mixed fractions (e.g. '1-1/2', '2-3/4')
    hyphen_frac_match = re.match(r'^(\d+)\s*-\s*(\d+)/(\d+)$', s)
    if hyphen_frac_match:
        whole = float(hyphen_frac_match.group(1))
        num = float(hyphen_frac_match.group(2))
        denom = float(hyphen_frac_match.group(3))
        if denom == 0:
            raise ValueError("Division by zero in fraction")
        return whole + (num / denom)

    # 3. Space-separated mixed numbers (e.g. '1 1/2', '2 1/4')
    space_frac_match = re.match(r'^(\d+)\s+(\d+)/(\d+)$', s)
    if space_frac_match:
        whole = float(space_frac_match.group(1))
        num = float(space_frac_match.group(2))
        denom = float(space_frac_match.group(3))
        if denom == 0:
            raise ValueError("Division by zero in fraction")
        return whole + (num / denom)

    # 4. Simple fractions (e.g. '1/2', '3/4')
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            denom = float(parts[1].strip())
            if denom == 0:
                raise ValueError("Division by zero in fraction")
            return float(parts[0].strip()) / denom

    # 5. Numeric ranges (e.g. '1-2', '1 to 2')
    range_match = re.match(r'^(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)$', s, re.IGNORECASE)
    if range_match:
        val1 = float(range_match.group(1))
        val2 = float(range_match.group(2))
        return (val1 + val2) / 2.0

    # 6. Standard float / integer string
    return float(s)


def format_culinary_fraction(q, unit=''):
    """
    Formats continuous scaled float into practical culinary fractions:
    1/8, 1/4, 1/3, 1/2, 2/3, 3/4, mixed numbers or rounded integers.
    """
    if q <= 0:
        return "0"

    # Quantities >= 10: round to whole number if close, else 1 decimal place
    if q >= 10:
        if abs(q - round(q)) < 0.15:
            return f"{int(round(q))}"
        return f"{round(q, 1)}".rstrip('0').rstrip('.')

    # Fractions < 1.15
    if q < 0.15:
        return "1/8"
    if 0.15 <= q < 0.29:
        return "1/4"
    if 0.29 <= q < 0.42:
        return "1/3"
    if 0.42 <= q < 0.60:
        return "1/2"
    if 0.60 <= q < 0.71:
        return "2/3"
    if 0.71 <= q < 0.88:
        return "3/4"
    if 0.88 <= q < 1.15:
        return "1"

    # For 1.15 <= q < 10: Mixed numbers
    whole = int(math.floor(q))
    frac = q - whole

    if frac < 0.125:
        return f"{whole}"
    elif 0.125 <= frac < 0.29:
        return f"{whole} 1/4"
    elif 0.29 <= frac < 0.42:
        return f"{whole} 1/3"
    elif 0.42 <= frac < 0.60:
        return f"{whole} 1/2"
    elif 0.60 <= frac < 0.71:
        return f"{whole} 2/3"
    elif 0.71 <= frac < 0.88:
        return f"{whole} 3/4"
    else:  # frac >= 0.88
        return f"{whole + 1}"


# ============================================================================
# COMPREHENSIVE 35+ CULINARY SUBSTITUTIONS DATABASE
# ============================================================================

EXPANDED_SUBSTITUTIONS = {
    # --- Dairy & Fats ---
    'butter': {
        'substitutes': ['Ghee (equal amount)', 'Olive oil (3/4 amount)', 'Coconut oil (1:1)', 'Vegan butter', 'Margarine'],
        'ratio': 'Use 3/4 cup oil per 1 cup butter',
        'tip': 'In baking, vegetable oil keeps cakes moist; in cooking, ghee provides higher smoke point and rich nutty flavor.',
        'diet': ['dairy-free', 'vegan']
    },
    'ghee': {
        'substitutes': ['Clarified butter', 'Coconut oil', 'Mustard oil (for pungent curries)', 'Sesame oil', 'Olive oil'],
        'ratio': '1:1 replacement',
        'tip': 'For authentic tadka, heat oil until hot before tempering seeds.',
        'diet': ['dairy-free', 'vegan']
    },
    'cream': {
        'substitutes': ['Greek yogurt whisked with whole milk', 'Whipped coconut cream', 'Soaked cashew paste with water', 'Silken tofu blended'],
        'ratio': '1:1 volume replacement',
        'tip': 'Cashew paste is the ideal vegan alternative for rich Indian curries (korma, makhani).',
        'diet': ['vegan', 'dairy-free']
    },
    'heavy cream': {
        'substitutes': ['Cashew cream (soaked cashews blended with water)', 'Coconut cream', 'Milk and melted butter whisked (3/4 cup milk + 1/4 cup butter)'],
        'ratio': '1:1 replacement',
        'tip': 'Cashew cream thickens naturally when simmered in soups and gravies.',
        'diet': ['vegan', 'dairy-free']
    },
    'sour cream': {
        'substitutes': ['Plain thick Greek yogurt (hung curd)', 'Cottage cheese blended with 1 tbsp lemon juice', 'Coconut cream with dash of lemon'],
        'ratio': '1:1 ratio',
        'tip': 'Greek yogurt matches acidity and fat percentage closely.',
        'diet': ['vegetarian']
    },
    'curd': {
        'substitutes': ['Plain yogurt', 'Greek yogurt diluted with 20% water', 'Buttermilk', 'Coconut yogurt'],
        'ratio': '1:1 ratio',
        'tip': 'Whisk thoroughly before adding to hot gravies to prevent splitting.',
        'diet': ['vegetarian']
    },
    'yogurt': {
        'substitutes': ['Curd', 'Greek yogurt', 'Coconut yogurt', 'Sour cream'],
        'ratio': '1:1 ratio',
        'tip': 'Whisk at room temperature to prevent curdling in warm dishes.',
        'diet': ['vegetarian', 'vegan']
    },
    'buttermilk': {
        'substitutes': ['Milk + 1 tbsp lemon juice or vinegar (rest 5 mins)', 'Diluted plain yogurt (1/2 cup yogurt + 1/2 cup water)'],
        'ratio': '1:1 ratio',
        'tip': 'The acid in buttermilk activates baking soda for fluffy pancakes and naan.',
        'diet': ['vegetarian']
    },
    'paneer': {
        'substitutes': ['Extra-firm pressed tofu', 'Halloumi cheese', 'Ricotta salata', 'Firm cottage cheese'],
        'ratio': '1:1 ratio by weight',
        'tip': 'Press tofu for 20 minutes and shallow fry before simmering in curry to absorb spices.',
        'diet': ['vegan', 'dairy-free']
    },
    'cheese': {
        'substitutes': ['Nutritional yeast (for cheesy flavor)', 'Vegan cheddar', 'Paneer (for mild firm texture)', 'Tofu crumbled'],
        'ratio': '1:1 replacement',
        'tip': 'Nutritional yeast provides savory umami without lactose.',
        'diet': ['vegan', 'dairy-free']
    },
    'milk': {
        'substitutes': ['Oat milk (neutral flavor)', 'Almond milk', 'Soy milk (high protein)', 'Coconut milk (for curries)'],
        'ratio': '1:1 replacement',
        'tip': 'Use unsweetened, unflavored plant milks for savory dishes.',
        'diet': ['vegan', 'dairy-free']
    },
    'coconut milk': {
        'substitutes': ['Cashew milk/paste with water', 'Heavy dairy cream', 'Almond milk with touch of coconut oil'],
        'ratio': '1:1 replacement',
        'tip': 'Cashew paste reproduces the rich body of coconut milk in South Indian dishes.',
        'diet': ['vegan']
    },
    'mayonnaise': {
        'substitutes': ['Greek yogurt', 'Mashed avocado with lemon juice', 'Hummus', 'Silken tofu blended with vinegar and mustard'],
        'ratio': '1:1 replacement',
        'tip': 'Avocado provides heart-healthy fats with creamy spreadability.',
        'diet': ['vegan', 'egg-free']
    },

    # --- Proteins ---
    'chicken': {
        'substitutes': ['Paneer cubes', 'Extra-firm tofu', 'Soya chunks / Nutrela (boiled & squeezed)', 'King oyster mushrooms', 'Green jackfruit (raw)'],
        'ratio': '1:1 by volume or weight',
        'tip': 'Soya chunks and raw green jackfruit replicate fibrous meat texture remarkably well.',
        'diet': ['vegetarian', 'vegan']
    },
    'beef': {
        'substitutes': ['Portobello mushrooms', 'Lentils (brown/green)', 'Black beans', 'Tempeh', 'Seitan'],
        'ratio': '1:1 by weight',
        'tip': 'Mushrooms provide rich savory glutamates matching the deep umami of beef.',
        'diet': ['vegetarian', 'vegan']
    },
    'pork': {
        'substitutes': ['Chicken thighs', 'Jackfruit (young green)', 'Extra-firm tofu smoked', 'Mushrooms'],
        'ratio': '1:1 replacement',
        'tip': 'Young jackfruit shreds like pulled pork when braised in spices.',
        'diet': ['vegetarian', 'halal']
    },
    'fish': {
        'substitutes': ['Firm tofu marinated in soy sauce and nori (seaweed)', 'Banana blossom', 'Hearts of palm'],
        'ratio': '1:1 replacement',
        'tip': 'Nori seaweed imparts an authentic seafood aroma to vegan proteins.',
        'diet': ['vegetarian', 'vegan']
    },
    'egg': {
        'substitutes': ['Flaxseed meal with water (1 tbsp ground flax + 3 tbsp water per egg)', 'Chia gel', 'Silken tofu (1/4 cup blended)', 'Mashed ripe banana (for baking)', 'Commercial egg replacer'],
        'ratio': '1 flax egg = 1 hen egg',
        'tip': 'Allow flaxseed mixture to rest for 5 minutes until gelatinous before folding into batter.',
        'diet': ['vegan', 'egg-free']
    },
    'eggs': {
        'substitutes': ['Flaxseed meal with water (1 tbsp : 3 tbsp per egg)', 'Chia gel', 'Silken tofu (1/4 cup per egg)', 'Aquafaba (3 tbsp per egg white)'],
        'ratio': '1 tbsp flax + 3 tbsp water per egg',
        'tip': 'Aquafaba (chickpea brine) whips into stiff peaks like egg whites.',
        'diet': ['vegan', 'egg-free']
    },
    'tofu': {
        'substitutes': ['Paneer', 'Tempeh', 'Seitan', 'Chickpea tofu (Shan tofu)'],
        'ratio': '1:1 replacement',
        'tip': 'Tempeh offers an earthier, firmer bite than tofu with probiotic benefits.',
        'diet': ['vegetarian']
    },
    'mutton': {
        'substitutes': ['Raw jackfruit (kathal)', 'Soya chaap', 'Portobello mushrooms', 'Lamb'],
        'ratio': '1:1 replacement',
        'tip': 'Slow-cook kathal with whole spices so it absorbs rich mutton-style gravy.',
        'diet': ['vegetarian', 'vegan']
    },

    # --- Flours & Thickeners ---
    'cornstarch': {
        'substitutes': ['Arrowroot starch (1:1)', 'All-purpose flour (use 2x amount)', 'Tapioca starch', 'Rice flour', 'Potato starch'],
        'ratio': '1 tbsp cornstarch = 2 tbsp all-purpose flour',
        'tip': 'Always dissolve in cold water before whisking into hot sauce to prevent lumps.',
        'diet': ['gluten-free']
    },
    'cornflour': {
        'substitutes': ['Arrowroot starch (1:1)', 'All-purpose flour (2x amount)', 'Tapioca starch', 'Rice flour'],
        'ratio': '1:1 with arrowroot, 1:2 with wheat flour',
        'tip': 'Slurry with cold water before stirring into hot simmering liquid.',
        'diet': ['gluten-free']
    },
    'besan': {
        'substitutes': ['Chickpea flour', 'Roasted gram powder (Sattu)', 'All-purpose flour mixed with pinch of turmeric', 'Oat flour'],
        'ratio': '1:1 replacement',
        'tip': 'Sattu requires less roasting time in kadhai compared to raw besan.',
        'diet': ['gluten-free']
    },
    'gram flour': {
        'substitutes': ['Chickpea flour', 'Besan', 'Yellow pea flour', 'Lentil flour'],
        'ratio': '1:1 replacement',
        'tip': 'Ensure flour is fresh and sifted to avoid bitterness.',
        'diet': ['gluten-free']
    },
    'maida': {
        'substitutes': ['Whole wheat flour / Atta (add 10% more liquid)', 'All-purpose gluten-free flour blend', 'Oat flour', 'Spelt flour'],
        'ratio': '1:1 volume',
        'tip': 'Atta yields denser dough with higher fiber content; allow 15 minutes resting.',
        'diet': ['higher-fiber']
    },
    'all-purpose flour': {
        'substitutes': ['Whole wheat flour (Atta)', '1:1 Gluten-free baking blend', 'Oat flour'],
        'ratio': '1:1 replacement (adjust liquid slightly for whole wheat)',
        'tip': 'For light cakes, sift flour twice before measuring.',
        'diet': ['vegetarian']
    },
    'atta': {
        'substitutes': ['Maida (all-purpose flour)', 'Multigrain flour blend', 'Spelt flour'],
        'ratio': '1:1 replacement',
        'tip': 'Atta absorbs more water than maida; knead with warm water for soft rotis.',
        'diet': ['vegetarian']
    },
    'rice flour': {
        'substitutes': ['Cornstarch', 'Tapioca flour', 'All-purpose flour', 'Millet flour'],
        'ratio': '1:1 replacement',
        'tip': 'Rice flour provides a super-crispy coating for pakoras and shallow-fried snacks.',
        'diet': ['gluten-free']
    },
    'breadcrumbs': {
        'substitutes': ['Crushed cornflakes', 'Rolled oats ground coarsely', 'Panko', 'Almond flour (low carb)'],
        'ratio': '1:1 replacement',
        'tip': 'Ground oats offer wholesome nutty crunch and high fiber.',
        'diet': ['gluten-free']
    },

    # --- Spices & Seasonings ---
    'garam masala': {
        'substitutes': ['Ground cumin, coriander, cinnamon and green cardamom blend', 'Curry powder', 'Allspice with black pepper'],
        'ratio': '1 tsp garam masala = 1/2 tsp cumin + 1/4 tsp coriander + 1/8 tsp cinnamon + 1/8 tsp cardamom',
        'tip': 'Add towards the end of cooking to preserve aromatic essential oils.',
        'diet': ['spice-blend']
    },
    'chaat masala': {
        'substitutes': ['Amchur (dry mango powder) + Kala namak (black salt) + Roasted cumin powder', 'Lemon juice with black salt'],
        'ratio': 'Equal parts amchur and cumin + pinch of black salt',
        'tip': 'Black salt provides the signature pungent sulfurous punch.',
        'diet': ['spice-blend']
    },
    'asafoetida': {
        'substitutes': ['Garlic powder mixed with onion powder (equal parts)', 'Finely minced leeks / shallots'],
        'ratio': 'Pinch of hing = 1/8 tsp garlic powder',
        'tip': 'Essential for allium-free Jain or Ayurvedic cooking.',
        'diet': ['allium-alternative']
    },
    'hing': {
        'substitutes': ['Garlic powder + onion powder (equal parts)', 'Finely minced shallots'],
        'ratio': '1 pinch hing = 1/8 tsp garlic powder',
        'tip': 'Bloom in hot ghee/oil at the start of cooking.',
        'diet': ['allium-alternative']
    },
    'amchur': {
        'substitutes': ['Fresh lemon or lime juice', 'Pomegranate seed powder (Anardana)', 'Tamarind paste diluted', 'Sumac'],
        'ratio': '1 tsp amchur = 1 tbsp fresh lemon juice',
        'tip': 'Add lemon juice at the end of cooking off heat to maintain freshness.',
        'diet': ['sour-agent']
    },
    'dry mango powder': {
        'substitutes': ['Amchur', 'Lemon juice', 'Tamarind paste', 'Anardana powder'],
        'ratio': '1 tsp = 1 tbsp lemon juice',
        'tip': 'Provides tart acidity without adding unwanted liquid to dry sabzis.',
        'diet': ['sour-agent']
    },
    'kasuri methi': {
        'substitutes': ['Dried celery leaves', 'Fenugreek seeds (ground very sparingly)', 'Fresh fenugreek leaves'],
        'ratio': '1 tbsp kasuri methi = 1 tbsp dried celery leaves',
        'tip': 'Rub kasuri methi between palms before sprinkling over gravies to release aromatics.',
        'diet': ['aromatic']
    },
    'cumin': {
        'substitutes': ['Ground caraway seeds', 'Fennel seeds (sweeter)', 'Ground coriander (milder)'],
        'ratio': '1:1 with caraway, or use 1/2 amount of fennel seeds',
        'tip': 'Caraway closely mirrors the warm earthy profile of cumin.',
        'diet': ['spice']
    },
    'coriander': {
        'substitutes': ['Cumin powder', 'Cardamom powder (sparingly)', 'Curry powder blend'],
        'ratio': '1 tsp coriander = 3/4 tsp cumin',
        'tip': 'Coriander seeds add floral, citrusy undertones to curry pastes.',
        'diet': ['spice']
    },
    'ginger': {
        'substitutes': ['Ground dry ginger powder (Sonth - 1/4 tsp per tbsp fresh)', 'Galangal', 'Grated candied ginger (for sweets)'],
        'ratio': '1 tbsp fresh ginger = 1/4 tsp dry ginger powder',
        'tip': 'Fresh ginger is punchier and aromatic, while dry ginger has a deeper, warmer heat.',
        'diet': ['fresh-spice']
    },
    'garlic': {
        'substitutes': ['Garlic powder (1/8 tsp per clove)', 'Minced shallots', 'Asafoetida / Hing (pinch in oil)'],
        'ratio': '1 clove garlic = 1/8 tsp garlic powder',
        'tip': 'Hing gives digestive benefits and garlic-onion aroma in Jain preparations.',
        'diet': ['allium']
    },
    'onion': {
        'substitutes': ['Shallots', 'Leeks', 'Spring onions (white part)', 'Asafoetida / Hing (pinch)', 'Fennel bulb'],
        'ratio': '1 medium onion = 3 small shallots or 1 cup chopped leeks',
        'tip': 'Leeks caramelize beautifully with a sweeter, gentler allium profile.',
        'diet': ['allium']
    },
    'turmeric': {
        'substitutes': ['Fresh grated turmeric root', 'Saffron threads (for vibrant yellow color)', 'Mild yellow mustard powder (sparingly)'],
        'ratio': '1/2 tsp dry turmeric = 1 tsp fresh grated root',
        'tip': 'Always pair turmeric with black pepper to boost curcumin bioavailability.',
        'diet': ['spice']
    },
    'cardamom': {
        'substitutes': ['Equal parts cinnamon and nutmeg blend', 'Allspice', 'Ginger powder'],
        'ratio': '1/2 tsp cardamom = 1/4 tsp cinnamon + 1/4 tsp nutmeg',
        'tip': 'Crush green pods right before using to retain volatile floral notes.',
        'diet': ['sweet-spice']
    },
    'cinnamon': {
        'substitutes': ['Nutmeg (use half amount)', 'Allspice', 'Cassia bark (1:1)', 'Cardamom'],
        'ratio': '1 tsp cinnamon = 1/2 tsp nutmeg',
        'tip': 'Cassia provides a sharper, spicier bark flavor common in biryanis.',
        'diet': ['sweet-spice']
    },
    'cloves': {
        'substitutes': ['Allspice (1:1)', 'Nutmeg ground', 'Cinnamon with black pepper'],
        'ratio': '1:1 with ground allspice',
        'tip': 'Cloves are intensely pungent; use whole buds sparingly in rice and curries.',
        'diet': ['spice']
    },
    'saffron': {
        'substitutes': ['Turmeric (for golden color, use 1/4 tsp)', 'Safflower petals', 'Annatto powder'],
        'ratio': 'Pinch saffron = 1/4 tsp turmeric',
        'tip': 'Turmeric mimics saffron\'s brilliant golden hue, but does not replicate its honey-floral perfume.',
        'diet': ['aromatic']
    },

    # --- Sweeteners & Acids ---
    'sugar': {
        'substitutes': ['Jaggery / Gur (grated)', 'Raw honey (use 3/4 amount)', 'Maple syrup', 'Coconut palm sugar', 'Stevia'],
        'ratio': '1 cup sugar = 3/4 cup honey or 1 cup grated jaggery',
        'tip': 'Jaggery adds earthy caramel depth to Indian sweets and dal.',
        'diet': ['natural-sweetener']
    },
    'brown sugar': {
        'substitutes': ['White sugar + 1 tbsp molasses per cup', 'Jaggery powder (Shakkar)', 'Coconut sugar'],
        'ratio': '1:1 replacement',
        'tip': 'Jaggery powder provides natural toffee notes perfect for baking and puddings.',
        'diet': ['natural-sweetener']
    },
    'honey': {
        'substitutes': ['Maple syrup', 'Agave nectar', 'Jaggery syrup', 'Date syrup'],
        'ratio': '1:1 replacement',
        'tip': 'Agave syrup dissolves easily in cold liquids and dressings.',
        'diet': ['vegan']
    },
    'tomato': {
        'substitutes': ['Tamarind pulp diluted with water', 'Canned crushed tomatoes', 'Roasted red bell pepper puree + lemon juice', 'Amchur powder'],
        'ratio': '1 cup tomato puree = 3/4 cup roasted bell pepper puree + 1 tbsp lemon juice',
        'tip': 'Bell pepper puree mimics natural sweetness and vibrant red color without nightshades.',
        'diet': ['nightshade-free']
    },
    'tomatoes': {
        'substitutes': ['Canned whole/crushed tomatoes', 'Roasted red bell pepper puree with lemon juice', 'Tamarind pulp', 'Tomato paste diluted with water'],
        'ratio': '2 fresh tomatoes = 1/2 cup canned tomato puree',
        'tip': 'Simmer bell pepper puree to match rich stew consistency.',
        'diet': ['nightshade-free']
    },
    'lemon juice': {
        'substitutes': ['Lime juice (1:1)', 'Apple cider vinegar (use 1/2 amount)', 'White wine vinegar', 'Amchur powder (1 tsp per tbsp)'],
        'ratio': '1 tbsp lemon juice = 1 tbsp lime juice or 1/2 tbsp vinegar',
        'tip': 'Lime juice is slightly more acidic; add in batches to taste.',
        'diet': ['acid']
    },
    'vinegar': {
        'substitutes': ['Lemon or lime juice (use double amount)', 'Apple cider vinegar', 'White wine vinegar', 'Tamarind water'],
        'ratio': '1 tbsp vinegar = 2 tbsp fresh lemon juice',
        'tip': 'Apple cider vinegar gives mild fruity acidity ideal for chutneys and dressings.',
        'diet': ['acid']
    },
    'soy sauce': {
        'substitutes': ['Tamari (gluten-free)', 'Coconut aminos (soy-free, sweeter)', 'Miso paste mixed with warm water', 'Worcestershire sauce'],
        'ratio': '1:1 with tamari, or 1:1 with coconut aminos (add pinch of salt)',
        'tip': 'Coconut aminos is naturally sweet and soy-free with 70% less sodium.',
        'diet': ['gluten-free', 'soy-free']
    },
    'tamarind': {
        'substitutes': ['Amchur powder mixed with warm water', 'Lemon juice mixed with brown sugar/jaggery', 'Pomegranate molasses', 'Kokum soaked'],
        'ratio': '1 tbsp tamarind pulp = 1 tbsp lemon juice + 1/2 tsp jaggery',
        'tip': 'Kokum provides a clean, tart tang without darkening light curries.',
        'diet': ['sour-agent']
    }
}


# ============================================================================
# RATVI ENGINE CLASS
# ============================================================================

class RatviEngine:
    """
    Ratvi — AI Culinary Companion Engine
    Dual-engine intelligence providing:
    1. Conversational intent understanding
    2. Dynamic serving-based non-linear ingredient and cooking time scaling
    3. 35+ item culinary substitutions with ratios and culinary advice
    4. Step-by-step guidance with timer extraction and navigation
    5. Hybrid fallback (deterministic offline <5ms with optional external LLM)
    """

    SUBSTITUTIONS = EXPANDED_SUBSTITUTIONS

    QUANTITY_REGEX = re.compile(
        r'^\s*([0-9\u00BC-\u00BE\u2150-\u215E]+(?:\s*-\s*[0-9\u00BC-\u00BE\u2150-\u215E]+/\d+|\s*-\s*\d+|\s+\d+/\d+|/\d+|\.\d+)?|\d+\.\d+)\s*'
        r'(cups?|c\.|tbsp?|tablespoons?|tsps?|teaspoons?|grams?|g|kg|ml|oz|ounces?|lbs?|pounds?|pinch(?:es)?|cloves?|cans?|cartons?|pkgs?|packages?|jars?|large|medium|small|whole|slices?|stalks?|bunches?|heads?|pieces?|bunch)?\s*(.*)$',
        re.IGNORECASE
    )

    def __init__(self, recipe_df=None):
        self.recipe_df = recipe_df

    def set_recipe_df(self, df):
        self.recipe_df = df

    parse_fraction = staticmethod(parse_culinary_fraction)

    @classmethod
    def get_scaling_exponent(cls, ingredient_line, ratio):
        """
        Determines power-law exponent alpha for ingredient line:
        - Bulk/base: alpha = 1.00
        - Liquids: alpha = 0.85 (upscale) / 0.92 (downscale)
        - Salt/sodium: alpha = 0.80 (upscale) / 0.95 (downscale)
        - Pungent spices: alpha = 0.80 (upscale) / 0.90 (downscale)
        - Cooking fats/oils: alpha = 0.75 (upscale) / 0.85 (downscale)
        - Leaveners: alpha = 0.80 (upscale) / 0.90 (downscale)
        - Aromatic spices: alpha = 0.90 (upscale) / 0.95 (downscale)
        """
        line_lower = ingredient_line.lower()
        is_upscaling = (ratio >= 1.0)

        # 1. Salt & Sodium
        if any(k in line_lower for k in ['salt', 'soy sauce', 'fish sauce', 'tamari', 'namak']):
            return 0.80 if is_upscaling else 0.95

        # 2. Pungent Spices
        if any(k in line_lower for k in [
            'red chilli', 'chilli powder', 'chili powder', 'cayenne', 'black pepper',
            'peppercorn', 'paprika', 'jalapeño', 'jalapeno', 'chilli', 'chili', 'clove'
        ]):
            return 0.80 if is_upscaling else 0.90

        # 3. Cooking Oils & Fats
        if any(k in line_lower for k in ['oil', 'ghee', 'butter', 'lard', 'shortening', 'fat']):
            return 0.75 if is_upscaling else 0.85

        # 4. Leaveners
        if any(k in line_lower for k in ['baking powder', 'baking soda', 'yeast']):
            return 0.80 if is_upscaling else 0.90

        # 5. Liquids & Broths
        if any(k in line_lower for k in ['water', 'stock', 'broth', 'milk', 'coconut milk', 'juice', 'soup']):
            return 0.85 if is_upscaling else 0.92

        # 6. Aromatic Spices
        if any(k in line_lower for k in [
            'cumin', 'coriander', 'turmeric', 'cardamom', 'cinnamon', 'fennel',
            'garam masala', 'hing', 'asafoetida', 'kasuri methi', 'fenugreek',
            'nutmeg', 'star anise', 'haldi', 'jeera', 'dhania', 'elaichi', 'dalchini'
        ]):
            return 0.90 if is_upscaling else 0.95

        # 7. Bulk Base
        return 1.00

    @classmethod
    def scale_ingredient(cls, ingredient_line, base_servings, target_servings):
        """
        Applies power-law formula Q_scaled = Q_base * (R ^ alpha) with culinary rounding.
        """
        if base_servings <= 0 or target_servings <= 0:
            return ingredient_line

        ratio = target_servings / base_servings
        line = ingredient_line.strip()

        match = cls.QUANTITY_REGEX.match(line)
        if match:
            num_str, unit, name = match.groups()
            try:
                base_qty = parse_culinary_fraction(num_str)
                alpha = cls.get_scaling_exponent(line, ratio)
                effective_ratio = (ratio ** alpha)
                scaled_qty = base_qty * effective_ratio

                formatted_qty = format_culinary_fraction(scaled_qty, unit or '')
                unit_str = f" {unit}" if unit else ""
                name_str = f" {name}".rstrip() if name else ""
                return f"{formatted_qty}{unit_str}{name_str}".strip()
            except Exception:
                return line
        return line

    def scale_recipe_servings(self, recipe_dict, target_servings):
        """
        Scales ingredients and non-linearly adjusts preparation and cooking times.
        Prep time scales as R^0.50, cook time scales as R^0.25.
        """
        base_servings = float(recipe_dict.get('Servings', 4) or 4)
        target_servings = float(target_servings)
        ratio = target_servings / base_servings if base_servings > 0 else 1.0

        raw_ingredients = recipe_dict.get('TranslatedIngredients') or recipe_dict.get('Ingredients') or ''
        if isinstance(raw_ingredients, str):
            s = raw_ingredients.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    ing_list = ast.literal_eval(s)
                except Exception:
                    ing_list = [i.strip() for i in s.split(',') if i.strip()]
            else:
                ing_list = [i.strip() for i in s.split(',') if i.strip()]
        elif isinstance(raw_ingredients, list):
            ing_list = raw_ingredients
        else:
            ing_list = []

        scaled_ingredients = [self.scale_ingredient(i, base_servings, target_servings) for i in ing_list]

        base_cook = float(recipe_dict.get('CookTimeInMins', 30) or 30)
        base_prep = float(recipe_dict.get('PrepTimeInMins', 15) or 15)

        adj_prep = max(1, int(round(base_prep * (ratio ** 0.50)))) if base_prep > 0 else 0
        adj_cook = max(1, int(round(base_cook * (ratio ** 0.25)))) if base_cook > 0 else 0
        total_time = adj_prep + adj_cook

        recipe_name = recipe_dict.get('RecipeName') or recipe_dict.get('TranslatedRecipeName') or recipe_dict.get('title') or 'Selected Recipe'
        recipe_id = recipe_dict.get('Srno')
        if recipe_id is not None:
            try:
                recipe_id = int(float(recipe_id))
            except Exception:
                pass

        return {
            'recipe_id': recipe_id,
            'recipe_name': recipe_name,
            'base_servings': int(round(base_servings)),
            'target_servings': int(round(target_servings)),
            'ratio': round(ratio, 2),
            'scaled_ingredients': scaled_ingredients,
            'prep_time_mins': adj_prep,
            'cook_time_mins': adj_cook,
            'total_time_mins': total_time,
            'note': f"Quantities calculated proportionally for {int(round(target_servings))} people with dampened spices and liquids for ideal flavor balance."
        }

    def find_substitutions(self, ingredient_name):
        """
        Finds culinary substitutes for an ingredient from 35+ items database.
        """
        query = ingredient_name.lower().strip()
        
        # 1. Direct key match
        if query in self.SUBSTITUTIONS:
            item = self.SUBSTITUTIONS[query]
            return {
                'ingredient': ingredient_name,
                'substitutes': item['substitutes'],
                'ratio': item.get('ratio', '1:1 ratio'),
                'tip': item.get('tip', f'Adjust seasoning according to taste when replacing {query}.'),
                'diet': item.get('diet', [])
            }

        # 2. Substring match
        for key, item in self.SUBSTITUTIONS.items():
            if key in query or query in key:
                return {
                    'ingredient': ingredient_name,
                    'substitutes': item['substitutes'],
                    'ratio': item.get('ratio', '1:1 ratio'),
                    'tip': item.get('tip', f'Adjust seasoning according to taste when replacing {key}.'),
                    'diet': item.get('diet', [])
                }

        # 3. Individual token match
        tokens = [w for w in re.split(r'\W+', query) if len(w) > 2]
        for token in tokens:
            if token in self.SUBSTITUTIONS:
                item = self.SUBSTITUTIONS[token]
                return {
                    'ingredient': ingredient_name,
                    'substitutes': item['substitutes'],
                    'ratio': item.get('ratio', '1:1 ratio'),
                    'tip': item.get('tip', f'Adjust seasoning according to taste when replacing {token}.'),
                    'diet': item.get('diet', [])
                }

        # 4. Graceful general fallback
        return {
            'ingredient': ingredient_name,
            'substitutes': [
                'A similar vegetable or protein with equivalent moisture/texture',
                'Neutral culinary oil or vegetable broth'
            ],
            'ratio': '1:1 replacement by volume or weight',
            'tip': 'For dry spices, use equivalent aromatic blends; for fresh ingredients, match water and fat content.',
            'diet': []
        }

    def extract_steps(self, recipe_dict):
        """
        Parses instructions list or string into individual clean steps.
        """
        raw_inst = recipe_dict.get('Instructions') or recipe_dict.get('TranslatedInstructions') or []
        if isinstance(raw_inst, str):
            s = raw_inst.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    inst_list = ast.literal_eval(s)
                except Exception:
                    inst_list = [step.strip() for step in s.split('.') if step.strip()]
            else:
                cleaned = re.sub(r'[\xa0\r]+', ' ', s)
                parts = re.split(r'(?:^|\n|\s+)(?:\d+[\.\)]\s+)', cleaned)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > 1:
                    inst_list = parts
                else:
                    inst_list = [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', cleaned) if p.strip()]
        elif isinstance(raw_inst, list):
            inst_list = [str(x).strip() for x in raw_inst if str(x).strip()]
        else:
            inst_list = []

        cleaned_steps = []
        for step in inst_list:
            clean = re.sub(r'^(?:step\s*\d+[:\.]?\s*|\d+[\.\)]\s*)', '', step, flags=re.IGNORECASE).strip()
            if clean:
                cleaned_steps.append(clean)
        return cleaned_steps if cleaned_steps else inst_list

    def extract_timers(self, text):
        """
        Extracts cooking durations from step instructions (e.g. '15 mins', '10 to 12 minutes').
        """
        matches = re.findall(
            r'(\d+(?:\s*(?:to|-)\s*\d+)?)\s*(mins?|minutes?|hours?|hrs?|seconds?|secs?)',
            text,
            re.IGNORECASE
        )
        return [f"{m[0].strip()} {m[1].strip()}" for m in matches]

    def _generate_suggestions(self, intent, recipe=None, sub_res=None):
        """Generates contextual suggestion chips for the UI."""
        if intent == 'servings_scaled':
            return ["Substitute for butter", "Show cooking steps", "Scale for 2 people"]
        elif intent == 'substitution':
            return ["Show cooking steps", "Scale for 4 people", "Replace another ingredient"]
        elif intent == 'instructions':
            return ["Next step", "Previous step", "Scale for 4 people"]
        elif recipe:
            return ["Scale for 6 people", "Substitute for butter", "Show cooking steps"]
        return ["Scale for 4 people", "Substitute for butter", "Show cooking steps"]

    def _process_deterministic(self, message, context=None, history=None, current_step=0):
        """
        High-performance rule-based culinary intelligence engine (<5ms).
        """
        msg = message.lower().strip()
        context = context or {}
        recipe = context.get('current_recipe')

        # 1. Serving-based portion scaling
        servings_match = re.search(r'(?:cook|prepare|scale|make|servings?|for)\s+(?:this\s+recipe\s+for\s+)?(\d+)\s*(?:people|persons?|servings?|guests?)?', msg)
        if servings_match and recipe:
            target_servings = int(servings_match.group(1))
            scaled = self.scale_recipe_servings(recipe, target_servings)
            ing_formatted = "\n".join([f"• {i}" for i in scaled['scaled_ingredients']])
            reply = (
                f"Here are the adjusted ingredients for **{scaled['recipe_name']}** scaled for **{scaled['target_servings']} people**:\n\n"
                f"{ing_formatted}\n\n"
                f"⏱ **Estimated Times**: ~{scaled['prep_time_mins']} mins prep, ~{scaled['cook_time_mins']} mins cook (~{scaled['total_time_mins']} mins total).\n\n"
                f"💡 *{scaled['note']}*"
            )
            return {
                'success': True,
                'reply': reply,
                'intent': 'servings_scaled',
                'type': 'servings_scaled',
                'engine_used': 'deterministic',
                'data': scaled,
                'suggestions': self._generate_suggestions('servings_scaled', recipe=recipe)
            }

        # 2. Culinary substitution
        sub_match = re.search(r'(?:substitute|replace|swap|alternative|instead of)\s+(?:for\s+|to\s+)?([a-z\s]+)', msg)
        if sub_match:
            ing_query = sub_match.group(1).strip()
            # Clean common trailing punctuation or stop words
            ing_query = re.sub(r'\b(with|something|else|in|recipe)\b', '', ing_query).strip()
            sub_res = self.find_substitutions(ing_query)
            alts_str = ", ".join(f"**{a}**" for a in sub_res['substitutes'])
            reply = (
                f"Looking to replace **{sub_res['ingredient']}**? Here are my top recommended substitutions:\n\n"
                f"👉 {alts_str}\n\n"
                f"⚖️ **Ratio**: {sub_res.get('ratio', '1:1 replacement')}\n\n"
                f"💡 *Culinary Tip*: {sub_res['tip']}"
            )
            return {
                'success': True,
                'reply': reply,
                'intent': 'substitution',
                'type': 'substitution',
                'engine_used': 'deterministic',
                'data': sub_res,
                'suggestions': self._generate_suggestions('substitution', sub_res=sub_res)
            }

        # 3. Step-by-step cooking instructions & navigation
        if any(w in msg for w in ['instruction', 'step', 'how to cook', 'how to make', 'guide me', 'steps', 'next', 'previous', 'back']):
            if recipe:
                steps = self.extract_steps(recipe)
                rec_name = recipe.get('RecipeName') or recipe.get('TranslatedRecipeName') or recipe.get('title') or 'this dish'
                
                # Check for explicit step progression
                if 'next' in msg:
                    target_step = current_step + 1
                elif any(w in msg for w in ['previous', 'back', 'prev']):
                    target_step = max(0, current_step - 1)
                else:
                    explicit_num = re.search(r'step\s*(\d+)', msg)
                    if explicit_num:
                        target_step = int(explicit_num.group(1)) - 1
                    else:
                        target_step = None

                if target_step is not None and 0 <= target_step < len(steps):
                    step_text = steps[target_step]
                    timers = self.extract_timers(step_text)
                    timer_note = f"\n\n⏱ **Active Timer**: {', '.join(timers)}" if timers else ""
                    reply = (
                        f"**Step {target_step + 1} of {len(steps)} for {rec_name}**:\n\n"
                        f"{step_text}{timer_note}\n\n"
                        f"Say *'Next'* to proceed or *'Back'* for the previous step."
                    )
                    return {
                        'success': True,
                        'reply': reply,
                        'intent': 'instructions',
                        'type': 'instructions',
                        'engine_used': 'deterministic',
                        'data': {
                            'step_number': target_step + 1,
                            'total_steps': len(steps),
                            'step_text': step_text,
                            'timers': timers,
                            'current_step': target_step
                        },
                        'suggestions': self._generate_suggestions('instructions', recipe=recipe)
                    }

                # Full overview or start of instructions
                formatted_steps = "\n".join([f"**Step {idx+1}**: {step}" for idx, step in enumerate(steps[:8])])
                reply = (
                    f"Here is your step-by-step cooking guide for **{rec_name}**:\n\n"
                    f"{formatted_steps}\n\n"
                    f"Need ingredient scaling or substitutions while cooking? Just ask!"
                )
                return {
                    'success': True,
                    'reply': reply,
                    'intent': 'instructions',
                    'type': 'instructions',
                    'engine_used': 'deterministic',
                    'data': {'steps': steps, 'current_step': 0},
                    'suggestions': self._generate_suggestions('instructions', recipe=recipe)
                }

        # 4. Shopping list / Meal planning
        if any(w in msg for w in ['shopping list', 'grocery', 'meal plan', 'plan my meals', 'weekly meals']):
            reply = (
                "To generate a consolidated shopping list:\n\n"
                "1. Pick the recipes you plan to cook this week.\n"
                "2. Tell me the number of people for each meal (e.g. *'Pasta for 4, Biryani for 6'*).\n"
                "3. I will consolidate all quantities into standardized units to prevent food waste."
            )
            return {'success': True, 'reply': reply, 'intent': 'shopping_list',
                    'type': 'general', 'engine_used': 'deterministic', 'data': {},
                    'suggestions': self._generate_suggestions('general_chat')}

        # 5. Nutrition / calories
        if any(w in msg for w in ['calorie', 'calories', 'protein', 'carb', 'nutrition', 'healthy', 'fat', 'fiber', 'kcal', 'diet']):
            if recipe:
                cal  = recipe.get('Calories (kcal)', 'N/A')
                prot = recipe.get('Protein (g)', 'N/A')
                carb = recipe.get('Carbohydrates (g)', 'N/A')
                fat  = recipe.get('Fats (g)', 'N/A')
                fib  = recipe.get('Fiber (g)', 'N/A')
                name = recipe.get('RecipeName', 'this recipe')
                reply = (
                    f"**Nutrition for {name}** (per serving):\n\n"
                    f"🔥 **Calories**: {cal} kcal\n"
                    f"💪 **Protein**: {prot} g\n"
                    f"🌾 **Carbs**: {carb} g\n"
                    f"🧈 **Fats**: {fat} g\n"
                    f"🌿 **Fiber**: {fib} g\n\n"
                    f"Diet tag: **{recipe.get('Diet', 'N/A')}**"
                )
            else:
                reply = (
                    "Open a specific recipe first, then ask me for its nutrition info!\n\n"
                    "I can show you exact calories, protein, carbs, fats and fiber per serving."
                )
            return {'success': True, 'reply': reply, 'intent': 'nutrition',
                    'type': 'nutrition', 'engine_used': 'deterministic', 'data': {},
                    'suggestions': ['Show cooking steps', 'Scale for 4 people', 'Substitute for butter']}

        # 6. Greeting
        if any(w in msg.split() for w in ['hello', 'hi', 'hey', 'hiya', 'namaste', 'morning', 'evening', 'sup', 'yo']) or msg in ['hi', 'hello', 'hey', 'namaste']:
            name = recipe.get('RecipeName', '') if recipe else ''
            ctx  = f" You're viewing **{name}** right now." if name else ''
            reply = (
                f"Hey there! 👩‍🍳 I'm **Ratvi**, your AI culinary companion!{ctx}\n\n"
                "I can help you:\n"
                "• **Scale portions** — *'Scale for 6 people'*\n"
                "• **Find substitutes** — *'Substitute for butter'*\n"
                "• **Walk through steps** — *'Show cooking steps'*\n"
                "• **Find recipes** — *'Biryani'* or *'Egg and Onion'*\n\n"
                "What would you like to cook today?"
            )
            return {'success': True, 'reply': reply, 'intent': 'greeting',
                    'type': 'greeting', 'engine_used': 'deterministic', 'data': {},
                    'suggestions': ['Scale for 4 people', 'Substitute for butter', 'Show cooking steps']}

        # 7. Thanks / praise
        if any(w in msg for w in ['thank', 'thanks', 'great', 'awesome', 'perfect', 'nice', 'good', 'helpful', 'love it']):
            reply = "You're welcome! Happy cooking! 🍳 Ask me anything else about ingredients, scaling, or cooking steps."
            return {'success': True, 'reply': reply, 'intent': 'thanks',
                    'type': 'thanks', 'engine_used': 'deterministic', 'data': {},
                    'suggestions': ['Scale for 4 people', 'Substitute for butter', 'Show cooking steps']}

        # 8. Recipe Search / Dish Query / Ingredient Query
        # Strip common search boilerplate
        clean_q = re.sub(
            r'^(?:find|search|show|suggest|give|recommend|what can i (?:cook|make)|recipe for|how (?:to cook|to make)|make me|cook|prepare)\s+',
            '', msg
        ).strip()
        clean_q = re.sub(r'\b(recipe|dish|please|quick|easy)\b', '', clean_q).strip()
        search_term = clean_q if clean_q else msg

        if self.recipe_df is not None and search_term and len(search_term) >= 2:
            # A. Search by Recipe Name
            name_matches = self.recipe_df[
                self.recipe_df['RecipeName'].str.contains(r'\b' + re.escape(search_term) + r'\b', case=False, na=False)
            ]
            if name_matches.empty:
                name_matches = self.recipe_df[
                    self.recipe_df['RecipeName'].str.contains(re.escape(search_term), case=False, na=False)
                ]

            # B. If not found by name, search by Ingredients
            if name_matches.empty:
                terms = [w.strip() for w in re.split(r'[,|and|\s]+', search_term) if len(w.strip()) > 2]
                if terms:
                    pattern = '.*'.join(re.escape(t) for t in terms[:3])
                    name_matches = self.recipe_df[
                        self.recipe_df['Ingredients_clean'].str.contains(pattern, case=False, na=False)
                    ]

            if not name_matches.empty:
                lines = []
                for _, r in name_matches.head(4).iterrows():
                    srno_id = int(float(r['Srno'])) if pd.notna(r.get('Srno')) else ''
                    link = f" [View Recipe](/recipe_details/{srno_id})" if srno_id else ""
                    lines.append(
                        f"• **{r['RecipeName']}** — ~{int(r.get('TotalTimeInMins', 30))} min, {r.get('Diet','')}{link}"
                    )
                reply = (
                    f"Here are my top recommended dishes for **{search_term}**:\n\n"
                    + "\n".join(lines)
                    + "\n\nClick any recipe to view complete ingredients, or ask me for cooking steps!"
                )
                return {
                    'success': True,
                    'reply': reply,
                    'intent': 'recipe_search',
                    'type': 'recipe_search',
                    'engine_used': 'deterministic',
                    'data': {'count': len(name_matches)},
                    'suggestions': ['Show cooking steps', 'Scale for 4 people', 'Substitute for butter']
                }

        # 9. Contextual fallback
        if recipe:
            name = recipe.get('RecipeName', 'your recipe')
            reply = (
                f"I'm here to help with **{name}**! 👩‍🍳\n\n"
                f"Try asking me:\n"
                f"• *'Scale for 5 people'* — adjust quantities for guests\n"
                f"• *'Substitute for cream'* — find ingredient swaps\n"
                f"• *'Show cooking steps'* — guided preparation instructions"
            )
        else:
            reply = (
                "I'm here to assist your cooking! 🍳 You can ask me:\n\n"
                "• **Find dishes**: *'Biryani'*, *'Butter Chicken'*, *'Pasta'*\n"
                "• **By ingredients**: *'Egg, onion, tomato'* or *'Chicken, rice'*\n"
                "• **Ingredient swaps**: *'Substitute for butter'*\n"
                "• **Portion scaling**: *'Scale for 6 people'*"
            )
        return {
            'success': True,
            'reply': reply,
            'intent': 'general_chat',
            'type': 'general',
            'engine_used': 'deterministic',
            'data': {},
            'suggestions': self._generate_suggestions('general_chat', recipe=recipe)
        }

    def _process_llm(self, message, context=None, history=None, current_step=0, api_key=None):
        """
        Optional external LLM adapter (Gemini API) with strict 3.0s timeout and context injection.
        """
        context = context or {}
        recipe = context.get('current_recipe')

        # If mathematical scaling, compute exact ground-truth values first
        servings_match = re.search(r'(?:cook|prepare|scale|make|servings?|for)\s+(\d+)', message, re.IGNORECASE)
        exact_scaled_data = None
        if servings_match and recipe:
            target_servings = int(servings_match.group(1))
            exact_scaled_data = self.scale_recipe_servings(recipe, target_servings)

        system_instruction = (
            "You are Ratvi, the friendly and knowledgeable AI Culinary Companion for the Harna recipe system. "
            "Deliver concise, warm, practical cooking assistance. Adhere strictly to the provided recipe context. "
            "Never hallucinate ingredient quantities."
        )

        prompt_parts = [system_instruction]
        if recipe:
            prompt_parts.append(f"Active Recipe: {recipe.get('RecipeName', 'Selected Recipe')}")
            if exact_scaled_data:
                prompt_parts.append(f"Ground-Truth Scaled Ingredients: {json.dumps(exact_scaled_data['scaled_ingredients'])}")

        prompt_parts.append(f"User: {message}")
        full_prompt = "\n\n".join(prompt_parts)

        # Gemini REST payload
        payload = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 400
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                result_json = json.loads(resp.read().decode('utf-8'))
                candidates = result_json.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts and 'text' in parts[0]:
                        reply_text = parts[0]['text'].strip()
                        intent = 'servings_scaled' if exact_scaled_data else 'general_chat'
                        return {
                            'success': True,
                            'reply': reply_text,
                            'intent': intent,
                            'type': intent,
                            'engine_used': 'llm',
                            'data': exact_scaled_data or {},
                            'suggestions': self._generate_suggestions(intent, recipe=recipe)
                        }

        raise RuntimeError("Empty response from LLM provider")

    def process_chat(self, message, context=None, history=None, current_step=0, options=None):
        """
        Public entrypoint supporting hybrid fallback:
        Attempts LLM if configured, seamlessly falling back to deterministic offline engine.
        """
        options = options or {}
        engine_choice = options.get('engine', 'hybrid')
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY')

        # If deterministic mode requested or no API key present:
        if engine_choice == 'deterministic' or not api_key:
            return self._process_deterministic(message, context=context, history=history, current_step=current_step)

        # Attempt LLM call with graceful exception fallback
        try:
            return self._process_llm(message, context=context, history=history, current_step=current_step, api_key=api_key)
        except Exception as e:
            logger.warning(f"[Ratvi Hybrid Fallback] LLM call failed ({e}). Reverting to Deterministic Engine.")
            result = self._process_deterministic(message, context=context, history=history, current_step=current_step)
            result['engine_used'] = 'deterministic'
            return result

    def process_message(self, message, context=None):
        """
        Backwards-compatible interface.
        """
        return self.process_chat(message, context=context)


# Global singleton instance
ratvi = RatviEngine()
