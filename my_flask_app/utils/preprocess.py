import nltk
import re
import ast
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

stopwords = set(nltk_stopwords.words('english'))
stemmer = PorterStemmer()

paren_pattern = re.compile(r'\(.*\)')
non_alpha_pattern = re.compile(r'[^a-z\s]+')

measurements = [
    'tbsp', 'tbsp.', 'tablespoon', 'tablespoons',
    'tsp', 'tsp.', 'teaspoon', 'teaspoons',
    'oz', 'ounce', 'ounces', 'ounc',
    'fl oz', 'fl. oz', 'fluid ounce', 'fluid ounces', 'fluid ounc',
    'c', 'cup', 'cups',
    'qt', 'quart', 'quarts',
    'pt', 'pint', 'pints','pinch'
    'gal', 'gallon', 'gallons',
    'lb', 'pound', 'pounds',
    'ml', 'mL', 'millilitre', 'milliliter', 'milliliters',
    'cc',
    'g', 'gram', 'grams',
    'kg', 'kilogram', 'kilograms',
    'l', 'L', 'liter', 'liters', 'litre','inch'
]

unwanted_words = [
    'chop', 'chopped', 'slice', 'sliced', 'cut', 'deseed', 'deseeded',
    'cooked', 'boil', 'boiled', 'soak', 'soaked', 'overnight', 'pressure',
    'fry', 'fried', 'grate', 'grated', 'puree', 'pureed', 'mince', 'minced',
    'dice', 'diced', 'shred', 'shredded','breast',
    'piece', 'pieces', 'piec',
    'large', 'medium', 'mediums', 'small', 'best', 'big',
    'light', 'lightly', 'little', 'medium-large', 'mini', 'thick','few',
    'drain', 'drained', 'rinsed', 'julienned', 'crush', 'crushed',
    'sifted', 'rough', 'roughly', 'finely', 'coarse', 'coarsely',
    'fresh', 'extra', 'extra-large', 'extra-small',
    'tiny','used','garnish','and','an','the','a','as','per','taste','or','whole','crushed','homemade','cleaned','washed','separated',
    'cubed','broken','bunch','dice','finely','from','grated','handful','heavy', 'hot', 'into', 'mashed', 'mix', 'moist', 'more', 'needed', 'optional', 'or', 'peeled', 'prepared','pureed','roasted','scooped','seasoned','seasoning','slightly','smashed','soaked','some','sprig','squeezed','stalk','stick','strained','thinly','toasted','tsp','warm','wedges','well','with','without','your','oneinch','long','split'
]

def preprocess_ingredients(ingredients_input):
    if not isinstance(ingredients_input, list):
        try:
            ingredients_list = ast.literal_eval(ingredients_input)
        except Exception:
            ingredients_list = ingredients_input.split(',')
    else:
        ingredients_list = ingredients_input

    preprocessed = []

    for ingredient in ingredients_list:
        ingredient = paren_pattern.sub('', ingredient).strip()
        ingredient = ingredient.split(',')[0].lower()
        ingredient = non_alpha_pattern.sub('', ingredient).strip()
        ingredient_tokens = word_tokenize(ingredient)
        stemmed_tokens = [stemmer.stem(token) for token in ingredient_tokens]
        filtered_tokens = [
            token for token in stemmed_tokens
            if token not in stopwords 
            and token not in measurements 
            and token not in unwanted_words 
            and len(token) > 1
        ]
        cleansed_ingredient = ' '.join(filtered_tokens)
        if ('oil' not in cleansed_ingredient and 
            'salt' not in cleansed_ingredient and 
            'water' not in cleansed_ingredient and 
            cleansed_ingredient != ''):
            preprocessed.append(cleansed_ingredient)
    
    return ' '.join(preprocessed)

def preprocess_user_ingredients(user_ingredients):
    ingredients_list = user_ingredients.split(',')
    return preprocess_ingredients(ingredients_list)

def preprocess_recipe_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z\s]', '', name)
    return name
