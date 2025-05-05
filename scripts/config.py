nutrient_units = {
    "sugar": "g",
    "fatty_acids_saturated": "g",
    "salt": "g",
    "fatty_acids_monounsaturated": "g",
    "fatty_acids_polyunsaturated": "g",
    "cholesterol": "mg",
    # "vitamin_a": "IU",
    "all_trans_retinol_equivalents_activity": "IU",
    "vitamin_c": "mg",
    "beta_carotene": "mcg",
    # "vitamin_e": "TAE",
    "vitamin_e_activity": "TAE",
    "vitamin_d": "mcg",
    "vitamin_k": "mcg",
    "thiamin": "mg",
    "riboflavin": "mg",
    "niacin": "mg",
    "vitamin_b6": "mg",
    "folate": "mcg",
    "vitamin_b12": "mcg",
    "calcium": "mg",
    "phosphorus": "mg",
    "magnesium": "mg",
    "iron": "mg",
    "zinc": "mg",
    "copper": "mg",
    "selenium": "mcg",
    "potassium": "mg",
    "sodium": "mg",
    "caffeine": "mg",
    "theobromine": "mg",
    "pantothenic_acid": "mg",
    "vitamin_b1": "mg",
    "vitamin_b2": "mg",
}
conversion_factors = {
    "mg": 1000,
    "g": 1,
    "mcg": 1000000,
    "IU": 1,
    "TAE": 1,
}

nutrient_info = {
    'folate_eaten':                                         {'unit': 'mcg', 'target': 400,  'type': 'qualifying'},
    'niacin_eaten':                                         {'unit': 'mg',  'target': 14,   'type': 'qualifying'},
    'pantothenic_acid_eaten':                               {'unit': 'mg',  'target': 5,    'type': 'qualifying'},
    'vitamin_b2_eaten':                                     {'unit': 'mg',  'target': 1.1,  'type': 'qualifying'},
    'vitamin_b1_eaten':                                     {'unit': 'mg',  'target': 1.2,  'type': 'qualifying'},
    'all_trans_retinol_equivalents_activity_eaten':         {'unit': 'IU', 'target': 700,  'type': 'qualifying'},
    'vitamin_b12_eaten':                                    {'unit': 'mcg', 'target': 2.4,  'type': 'qualifying'},
    'vitamin_b6_eaten':                                     {'unit': 'mg',  'target': 1.3,  'type': 'qualifying'},
    'vitamin_c_eaten':                                      {'unit': 'mg',  'target': 75,   'type': 'qualifying'},
    'vitamin_d_eaten':                                      {'unit': 'mcg', 'target': 7.5,  'type': 'qualifying'},
    'vitamin_e_activity_eaten':                             {'unit': 'TAE', 'target': 15,    'type': 'qualifying'},
    'calcium_eaten':                                        {'unit': 'mg',  'target': 1000, 'type': 'qualifying'},
    'iron_eaten':                                           {'unit': 'mg',  'target': 18,   'type': 'qualifying'},
    'magnesium_eaten':                                      {'unit': 'mg',  'target': 320,  'type': 'qualifying'},
    'phosphorus_eaten':                                     {'unit': 'mg',  'target': 700,  'type': 'qualifying'},
    'potassium_eaten':                                      {'unit': 'mg',  'target': 4700, 'type': 'qualifying'},
    'zinc_eaten':                                           {'unit': 'mg',  'target': 8,    'type': 'qualifying'},
    'fiber_eaten':                                          {'unit': 'g',   'target': 25,   'type': 'qualifying'},
    'protein_eaten':                                        {'unit': 'g',   'target': 46,   'type': 'qualifying'},
    'water_eaten':                                          {'unit': 'g',   'target' : 2700, 'type' : 'qualifying'}, 
    'fat_eaten':                                            {'unit': 'g',   'target': 78,   'type': 'disqualifying'},
    'fatty_acids_saturated_eaten':                          {'unit': 'g',   'target': 22,   'type': 'disqualifying'},
    'cholesterol_eaten':                                    {'unit': 'mg',  'target': 300,  'type': 'disqualifying'},
    'sugar_eaten':                                          {'unit': 'g',   'target': 125,  'type': 'disqualifying'},
    'sodium_eaten':                                         {'unit': 'mg',  'target': 2400, 'type': 'disqualifying'},
    'salt_eaten' :                                          {'unit' : 'g',  'target': 6,   'type': 'disqualifying'}
}

nutrient_choice = ['folate_eaten', 
                   'niacin_eaten', 
                   'pantothenic_acid_eaten', 
                   'vitamin_b2_eaten', 
                   'vitamin_b1_eaten', 
                   'all_trans_retinol_equivalents_activity_eaten', 
                   'beta_carotene_eaten', 
                   'vitamin_b12_eaten', 
                   'vitamin_b6_eaten', 
                   'vitamin_c_eaten', 
                   'vitamin_d_eaten', 
                   'vitamin_e_activity_eaten', 
                   'calcium_eaten', 
                   'iron_eaten', 
                   'magnesium_eaten', 
                   'phosphorus_eaten', 
                   'potassium_eaten', 
                   'zinc_eaten', 
                   'fiber_eaten', 
                   'protein_eaten']

fruits_categories = {
    'compotes' : [
        'compote', 'mus', 'fruchtmus'
    ],
    'citrus' : [
        'lemon', 'citron', 'zitrone',
        'orange',
        'grapefruit',
        'mandarine',
        'clementine'
    ],
    'berries' : [
        'berry', 'baie', 'beere',
        'rasberry', 'framboise', 'himbeere', 
        'strawberry', 'fraise', 'erdbeere',
        'blackberry', 'mûres', 'brombeere'
    ], 
    'tropical' : [
        'mango', 'mangue',
        'papaya',
        'kiwi',
        'passion',
        'litchi'
    ]
}

vegetable_categories = {
    'root_vegetables': [
        'carrot', 'carotte', 'karotte', 'möhre',
        'beetroot', 'betterave', 'rote bete', 'rübe',
        'turnip', 'navet', 'steckrübe',
        'parsnip', 'panais', 'pastinake',
        'celeriac', 'céleri-rave', 'knollensellerie',
        'onion', 'oignon', 'zwiebel',
        'radish', 'radis', 'radieschen',
        'fennel', 'fenouil', 'fenchel',
        'potato', 'pomme de terre', 'kartoffel'
    ],
    'leafy_vegetables': [
        'spinach', 'épinard', 'spinat',
        'beet', 'blette', 'mangold',
        'chicory', 'chicorée',
        'endive', 'endivie',
        'lettuce', 'laitue', 'kopfsalat', 'salat',
        'kale', 'chou frisé', 'grünkohl',
        'cabbage', 'chou', 'kohl',
        'rocket', 'roquette', 'rucola',
        'arugula', 
        'mustard', 'moutarde', 'senf'
    ],
    'fruity_vegetables': [
        'avocado', 'avocat',
        'asparagus', 'asperge', 'spargel',
        'cucumber', 'cucumbers', 'concombre',
        'eggplant', 'aubergine',
        'olive',  
        'pea', 'peas', 'pois', 'erbse', 'erbsen',
        'maïs', 'mais',
        'tomato', 'tomate',
        'pepper', 'poivron', 'paprika',
        'squash', 'courge', 'kürbis',
        'zucchini', 'courgette'
    ],
    'pickled_vegetables' : [
        'cornichon', 'chirat',
        'pickled', 'mariné', 'eingelegt'

    ]
}


meal_time_columns = {
    'breakfast': 'breakfast_time',
    'lunch': 'lunch_time',
    'snack': 'snack_time',
    'dinner': 'dinner_time'
}



