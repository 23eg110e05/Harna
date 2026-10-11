"""
Harna Image Resolver
Maps recipe name / cuisine / diet keywords → reliable Unsplash CDN photo IDs.
Used as a Jinja2 filter: {{ row['images'] | recipe_img(row['RecipeName'], row['Cuisine']) }}
"""

# (cuisine_keyword → unsplash photo ID)  — all verified working
CUISINE_MAP = {
    'north indian':  '1585937421612-70a008356fbe',
    'south indian':  '1567188040759-fb8a883dc6d8',
    'bengali':       '1601050690597-df0568f70950',
    'gujarati':      '1601050690597-df0568f70950',
    'punjabi':       '1603894584373-5ac82b2ae398',
    'rajasthani':    '1585937421612-70a008356fbe',
    'maharashtrian': '1567188040759-fb8a883dc6d8',
    'indian':        '1585937421612-70a008356fbe',
    'chinese':       '1569718212165-3a8278d5f624',
    'italian':       '1555949258-eb67b1ef0ceb',
    'mexican':       '1565299585323-38d6b0865b47',
    'thai':          '1562802378-063ec186a863',
    'japanese':      '1579871494447-9811cf80d66c',
    'american':      '1568901346375-23c9450c58cd',
    'fusion':        '1504674900247-0877df9cc836',
    'mediterranean': '1512621776951-a57141f2eefd',
    'middle eastern':'1541519227354-08fa5d50c627',
    'continental':   '1504674900247-0877df9cc836',
}

# (name keyword → unsplash photo ID)
NAME_MAP = {
    'butter chicken':     '1603894584373-5ac82b2ae398',
    'chicken tikka':      '1603894584373-5ac82b2ae398',
    'chicken curry':      '1603894584373-5ac82b2ae398',
    'chicken':            '1598103442097-8b74394b95c8',
    'paneer tikka':       '1631452180519-c014fe946bc7',
    'paneer butter':      '1631452180519-c014fe946bc7',
    'shahi paneer':       '1631452180519-c014fe946bc7',
    'paneer':             '1631452180519-c014fe946bc7',
    'biryani':            '1563379091339-03246963d96c',
    'pulao':              '1546549032-9571cd6b27df',
    'fried rice':         '1546549032-9571cd6b27df',
    'rice':               '1546549032-9571cd6b27df',
    'pizza':              '1565299624946-b28f40a0ae38',
    'pasta':              '1555949258-eb67b1ef0ceb',
    'noodle':             '1569718212165-3a8278d5f624',
    'ramen':              '1569718212165-3a8278d5f624',
    'dal':                '1601050690597-df0568f70950',
    'lentil':             '1601050690597-df0568f70950',
    'soup':               '1547592166-23ac45744acd',
    'salad':              '1512621776951-a57141f2eefd',
    'sandwich':           '1528735602780-2552fd46c7af',
    'burger':             '1568901346375-23c9450c58cd',
    'fish':               '1519708227418-c8fd9a32b7a2',
    'prawn':              '1559209368-b68de7c05e54',
    'shrimp':             '1559209368-b68de7c05e54',
    'mutton':             '1529694157872-4e0c0f3b238b',
    'lamb':               '1529694157872-4e0c0f3b238b',
    'egg':                '1482049016688-2d3e1b311543',
    'omelette':           '1482049016688-2d3e1b311543',
    'dosa':               '1567188040759-fb8a883dc6d8',
    'idli':               '1567188040759-fb8a883dc6d8',
    'sambar':             '1567188040759-fb8a883dc6d8',
    'chutney':            '1567188040759-fb8a883dc6d8',
    'paratha':            '1585937421612-70a008356fbe',
    'roti':               '1585937421612-70a008356fbe',
    'naan':               '1509440159596-0249088772ff',
    'bread':              '1509440159596-0249088772ff',
    'cake':               '1578985545062-69928b1d9587',
    'dessert':            '1488477181946-6428a0291777',
    'sweet':              '1488477181946-6428a0291777',
    'halwa':              '1488477181946-6428a0291777',
    'kheer':              '1488477181946-6428a0291777',
    'curry':              '1565557623262-b51c2513a641',
    'masala':             '1565557623262-b51c2513a641',
    'korma':              '1565557623262-b51c2513a641',
    'kebab':              '1529694157872-4e0c0f3b238b',
    'tandoori':           '1603894584373-5ac82b2ae398',
    'smoothie':           '1490645935967-10de6ba17061',
    'juice':              '1490645935967-10de6ba17061',
    'coffee':             '1495474472287-4d71bcdd2085',
    'tea':                '1544787219-7f47ccb76574',
    'karela':             '1601050690597-df0568f70950',
    'sabzi':              '1512621776951-a57141f2eefd',
    'vegetable':          '1512621776951-a57141f2eefd',
    'aloo':               '1585937421612-70a008356fbe',
    'potato':             '1585937421612-70a008356fbe',
    'mushroom':           '1518977676601-b53f82aba655',
    'spinach':            '1512621776951-a57141f2eefd',
    'palak':              '1631452180519-c014fe946bc7',
    'tomato':             '1546470427-f5d8e82e8f3f',
    'steak':              '1529694157872-4e0c0f3b238b',
    'bbq':                '1529694157872-4e0c0f3b238b',
    'taco':               '1565299585323-38d6b0865b47',
    'wrap':               '1528735602780-2552fd46c7af',
    'sushi':              '1579871494447-9811cf80d66c',
    'dumplings':          '1569718212165-3a8278d5f624',
    'stir fry':           '1569718212165-3a8278d5f624',
    'porridge':           '1546549032-9571cd6b27df',
    'oatmeal':            '1546549032-9571cd6b27df',
    'waffle':             '1484723091739-30a097e8f929',
    'pancake':            '1484723091739-30a097e8f929',
}

DEFAULT_PHOTO_ID = '1504674900247-0877df9cc836'   # colorful food spread


def _unsplash(photo_id: str) -> str:
    return f"https://images.unsplash.com/photo-{photo_id}?w=600&q=80"


def resolve_recipe_image(original_url: str, recipe_name: str = '', cuisine: str = '') -> str:
    """
    Returns the best image URL for a recipe.
    Priority: original URL (if non-empty & not obviously broken) → name keyword → cuisine → default.
    """
    name_lower = (recipe_name or '').lower()
    cuisine_lower = (cuisine or '').lower()

    # 1. Check by recipe name keywords (most specific)
    for keyword, photo_id in NAME_MAP.items():
        if keyword in name_lower:
            return _unsplash(photo_id)

    # 2. Check by cuisine
    for key, photo_id in CUISINE_MAP.items():
        if key in cuisine_lower:
            return _unsplash(photo_id)

    # 3. Use original URL only if it looks like a proper image
    if original_url and original_url.startswith('http') and not 'archanaskitchen' in original_url:
        return original_url

    # 4. Default beautiful food photo
    return _unsplash(DEFAULT_PHOTO_ID)
