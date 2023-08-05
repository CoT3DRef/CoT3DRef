from collections import defaultdict

shape_words = {
    'thin', 'thinner', 'thinnest',
    'fat', 'fatter', 'fattest',
    'large', 'larger', 'largest',
    'small', 'smaller', 'smallest',
    'wide', 'wider', 'widest',
    'long', 'longer', 'longest',
    'horizontal', 'vertical',
    'cube', 'cuboid', 'cubic', 'cylinder', 'cylindrical',
    'tall', 'taller', 'tallest',
    'thick', 'thicker', 'thickest',
    'narrow', 'narrower', 'narrowest',
    'skinny', 'skinnier', 'skinniest',
    'lump', 'angle',  'point', 'pointy',
    'plane', 'planar', 'surface', 'pyramid',
    'circle', 'circular',
    'triangle', 'triangular',
    'cone', 'conical',
    'rectangle', 'rectangular',
    'sphere', 'spherical',
    'ellipse', 'ellipsoid',
    'elliptical', 'cylindrical',
    'symmetry', 'symmetric', 'symmetrical',
    'curve', 'curvy', 'curvier', 'curviest',
    'shallow', 'shallower', 'shallowest',
    'deep', 'deeper', 'deepest',
    'flat', 'flatter', 'flattest',
    'oval',  'semicircle', 'square', 'squarish',
    'pentagon', 'hexagon',
    'octagon', 'parallelogram', 'quadrilateral',
    'rhombus', 'polygon'}

# Note the word 'corner' can be also used to refer to a shape. We opt adding to the spatial words
# since this is a much more common use in NR3D.

spatial_prepositions = {'aboard', 'above', 'across', 'adjacent', 'against', 'ahead', 'along',
                        'alongside', 'amid', 'amidst', 'among', 'amongst', 'apart', 'around',
                        'aside', 'astride', 'at', 'away', 'behind', 'below', 'beneath', 'beside',
                        'between', 'beyond', 'by', 'down', 'inside', 'into',
                        'near', 'nearby', 'on', 'onto', 'opposite', 'out', 'outside', 'over',
                        'through', 'together', 'toward', 'under', 'underneath', 'up', 'upper', 'within'}
# left out: 'about', 'in', 'round'

spatial_words = {'far', 'farthest', 'furthest', 'nearest',
                 'holding', 'holds', 'supporting', 'supports',
                 'left', 'right', 'front', 'side',
                 'low', 'lower', 'lowest',
                 'center', 'corner', 'middle', 'closest'}

spatial_tokens = spatial_words.union(spatial_prepositions)
spatial_expressions = [['close', 'to'], ['next', 'to'], ['back', 'of']]


# P will put below in a nicer way.
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
            matches.append(pattern)
    return matches


def uses_spatial_reasoning(tokens):
    exact_word = sum([i in tokens for i in spatial_tokens]) > 0  # at least one word (exact match)
    if exact_word:
        return True
    for s in spatial_expressions:
        if len(subfinder(tokens, s)) > 0:
            return True
    return False


allocentric_words = {'across', 'behind', 'corner', 'left', 'right', 'front', 'back'}

object_words = {'air hockey table', 'airplane', 'alarm', 'alarm clock', 'armchair', 'arm chair', 'baby mobile',
                'backpack', 'bag',
                'bag of coffee beans', 'ball', 'banana holder', 'bananas', 'banister', 'banner', 'bar', 'barricade',
                'basket', 'bath products', 'bath walls', 'bathrobe', 'bathroom cabinet', 'bathroom counter',
                'bathroom stall', 'bathroom stall door', 'bathroom vanity', 'bathtub', 'battery disposal jar',
                'beachball', 'beanbag chair', 'bear', 'bed', 'beer bottles', 'bench', 'bicycle', 'bike lock',
                'bike pump', 'bin', 'blackboard', 'blanket', 'blinds', 'block', 'board', 'boards', 'boat', 'boiler',
                'book', 'book rack', 'books', 'bookshelf', 'bookshelves', 'boots', 'bottle', 'bowl', 'box', 'boxes',
                'boxes of paper', 'breakfast bar', 'briefcase', 'broom', 'bucket', 'bulletin board', 'bunk bed',
                'cabinet', 'cabinet door', 'cabinet doors', 'cabinets', 'cable', 'calendar', 'camera', 'can', 'candle',
                'canopy', 'car', 'card', 'cardboard', 'carpet', 'carseat', 'cart', 'carton', 'case',
                'case of water bottles', 'cat litter box', 'cd case', 'ceiling', 'ceiling fan', 'ceiling light',
                'chain', 'chair', 'chandelier', 'changing station', 'chest', 'clock', 'closet', 'closet ceiling',
                'closet door', 'closet doorframe', 'closet doors', 'closet floor', 'closet rod', 'closet shelf',
                'closet wall', 'closet walls', 'cloth', 'clothes', 'clothes dryer', 'clothes dryers', 'clothes hanger',
                'clothes hangers', 'clothing', 'clothing rack', 'coat', 'coat rack', 'coatrack', 'coffee box',
                'coffee kettle', 'coffee maker', 'coffee table', 'column', 'compost bin', 'computer tower',
                'conditioner bottle', 'container', 'controller', 'cooking pan', 'cooking pot', 'copier', 'costume',
                'couch', 'couch cushions', 'counter', 'covered box', 'crate', 'crib', 'cup', 'cups', 'curtain',
                'curtains', 'cushion', 'cutting board', 'dart board', 'decoration', 'desk', 'desk lamp', 'diaper bin',
                'dining table', 'dish rack', 'dishwasher', 'dishwashing soap bottle', 'dispenser', 'display',
                'display case', 'display rack', 'divider', 'doll', 'dollhouse', 'dolly', 'door', 'doorframe', 'doors',
                'drawer', 'dress rack', 'dresser', 'drum set', 'dryer sheets', 'drying rack', 'duffel bag', 'dumbbell',
                'dustpan', 'easel', 'electric panel', 'elevator', 'elevator button', 'elliptical machine', 'end table',
                'envelope', 'exercise bike', 'exercise machine', 'exit sign', 'fan', 'faucet', 'file cabinet',
                'fire alarm', 'fire extinguisher', 'fireplace', 'flag', 'flip flops', 'floor', 'flower stand',
                'flowerpot', 'folded chair', 'folded chairs', 'folded ladder', 'folded table', 'folder', 'food bag',
                'food container', 'food display', 'foosball table', 'footrest', 'footstool', 'frame', 'frying pan',
                'furnace', 'furniture', 'fuse box', 'futon', 'garage door', 'garbage bag', 'glass doors', 'globe',
                'golf bag', 'grab bar', 'grocery bag', 'guitar', 'guitar case', 'hair brush', 'hair dryer', 'hamper',
                'hand dryer', 'hand rail', 'hand sanitzer dispenser', 'hand towel', 'handicap bar', 'handrail',
                'hanging', 'hat', 'hatrack', 'headboard', 'headphones', 'heater', 'helmet', 'hose', 'hoverboard',
                'humidifier', 'ikea bag', 'instrument case', 'ipad', 'iron', 'ironing board', 'jacket', 'jar', 'kettle',
                'keyboard', 'keyboard piano', 'kitchen apron', 'kitchen cabinet', 'kitchen cabinets', 'kitchen counter',
                'kitchen island', 'kitchenaid mixer', 'knife block', 'ladder', 'lamp', 'lamp base', 'laptop',
                'laundry bag', 'laundry basket', 'laundry detergent', 'laundry hamper', 'ledge', 'legs', 'light',
                'light switch', 'loft bed', 'loofa', 'luggage', 'luggage rack', 'luggage stand', 'lunch box', 'machine',
                'magazine', 'magazine rack', 'mail', 'mail tray', 'mailbox', 'mailboxes', 'map', 'massage chair', 'mat',
                'mattress', 'medal', 'messenger bag', 'metronome', 'microwave', 'mini fridge', 'mirror', 'mirror doors',
                'monitor', 'mouse', 'mouthwash bottle', 'mug', 'music book', 'music stand', 'nerf gun', 'night lamp',
                'nightstand', 'notepad', 'object', 'office chair', 'open kitchen cabinet', 'organizer',
                'organizer shelf', 'ottoman', 'oven', 'oven mitt', 'painting', 'pantry shelf', 'pantry wall',
                'pantry walls', 'pants', 'paper', 'paper bag', 'paper cutter', 'paper organizer', 'paper towel',
                'paper towel dispenser', 'paper towel roll', 'paper tray', 'papers', 'person', 'photo', 'piano',
                'piano bench', 'picture', 'pictures', 'pillar', 'pillow', 'pillows', 'ping pong table', 'pipe', 'pipes',
                'pitcher', 'pizza boxes', 'plant', 'plastic bin', 'plastic container', 'plastic containers',
                'plastic storage bin', 'plate', 'plates', 'plunger', 'podium', 'pool table', 'poster', 'poster cutter',
                'poster printer', 'poster tube', 'pot', 'potted plant', 'power outlet', 'power strip', 'printer',
                'projector', 'projector screen', 'purse', 'quadcopter', 'rack', 'rack stand', 'radiator', 'rail',
                'railing', 'range hood', 'recliner chair', 'recycling bin', 'refrigerator', 'remote', 'rice cooker',
                'rod', 'rolled poster', 'roomba', 'rope', 'round table', 'rug', 'salt', 'santa', 'scale', 'scanner',
                'screen', 'seat', 'seating', 'sewing machine', 'shampoo', 'shampoo bottle', 'shelf', 'shirt', 'shoe',
                'shoe rack', 'shoes', 'shopping bag', 'shorts', 'shower', 'shower control valve', 'shower curtain',
                'shower curtain rod', 'shower door', 'shower doors', 'shower floor', 'shower head', 'shower wall',
                'shower walls', 'shredder', 'sign', 'sink', 'sliding wood door', 'slippers', 'smoke detector', 'soap',
                'soap bottle', 'soap dish', 'soap dispenser', 'sock', 'soda stream', 'sofa bed', 'sofa chair',
                'speaker', 'sponge', 'spray bottle', 'stack of chairs', 'stack of cups', 'stack of folded chairs',
                'stair', 'stair rail', 'staircase', 'stairs', 'stand', 'stapler', 'starbucks cup', 'statue', 'step',
                'step stool', 'sticker', 'stool', 'storage bin', 'storage box', 'storage container',
                'storage organizer', 'storage shelf', 'stove', 'structure', 'studio light', 'stuffed animal',
                'suitcase', 'suitcases', 'sweater', 'swiffer', 'switch', 'table', 'tank', 'tap', 'tape', 'tea kettle',
                'teapot', 'teddy bear', 'telephone', 'telescope', 'thermostat', 'tire', 'tissue box', 'toaster',
                'toaster oven', 'toilet', 'toilet brush', 'toilet flush button', 'toilet paper',
                'toilet paper dispenser', 'toilet paper holder', 'toilet paper package', 'toilet paper rolls',
                'toilet seat cover dispenser', 'toiletry', 'toolbox', 'toothbrush', 'toothpaste', 'towel', 'towel rack',
                'towels', 'toy dinosaur', 'toy piano', 'traffic cone', 'trash bag', 'trash bin', 'trash cabinet',
                'trash can', 'tray', 'tray rack', 'treadmill', 'tripod', 'trolley', 'trunk', 'tube', 'tupperware', 'tv',
                'tv stand', 'umbrella', 'urinal', 'vacuum cleaner', 'vase', 'vending machine', 'vent', 'wall',
                'wall hanging', 'wall lamp', 'wall mounted coat rack', 'wardrobe', 'wardrobe cabinet',
                'wardrobe closet', 'washcloth', 'washing machine', 'washing machines', 'water bottle', 'water cooler',
                'water fountain', 'water heater', 'water pitcher', 'wet floor sign', 'wheel', 'whiteboard',
                'whiteboard eraser', 'window', 'windowsill', 'wood', 'wood beam', 'workbench',
                'yoga mat'}

hacked_rules = {
    'bottle': ['detergent'],
    'bench': 'table',
    'cabinet': ['shelf'],
}

instance_syn = defaultdict(list)
instance_syn.update({
    'armchair': ['arm chair'],
    'backpack': ['back pack'],
    'bag': ['purse', 'traveling bag', 'travel bag', 'luggage'],
    'bar': ['rail', 'fixture', 'ledge'],
    'bathroom stall': ['stall'],
    'bathroom stall door': ['stall door'],
    'bed': [],
    'bench': [],
    'blackboard': [],
    'blanket': [],
    'board': ['chart', 'chalkboard', 'chalk board', 'black board'],
    'bookshelf': ['bookcase', 'book shelf'],
    'bottle': ['jug'],
    'box': [],
    'bin': [],
    'cabinet': ['sideboard', 'locker', 'drawers'],
    'cart': ['wagon'],
    'chair': ['seat', 'barstool', 'seating'],
    'coffee maker': ['coffee machine', 'machine', 'coffeemaker'],
    'coffee table': [],
    'computer tower': ['pc tower', 'tower', 'cpu'],
    'computer': ['pc', 'desktop'],
    'couch': ['sofa', 'loveseat', 'love seat'],
    'cup': ['holder', 'mug', 'tea pot'],
    'curtain': [],
    'desk': [],
    'door': ['entrance'],
    'doors': [],
    'dresser': [],
    'end table': [],
    'file cabinet': [],
    'keyboard': [],
    'kitchen cabinet': ['cupboard', 'cup board', 'kitchen drawers', 'kitchen unit'],
    'kitchen counter': ['counter'],
    'lamp': [],
    'laptop': [],
    'laundry hamper': ['hamper', 'basket'],
    'light': [],
    'microwave': [],
    'mirror': [],
    'monitor': ['screen'],
    'mouse': [],
    'nightstand': ['night stand'],
    'office chair': ['desk chair'],
    'ottoman': [],
    'oven': [],
    'paper towel dispenser': ['dispenser'],
    'person': ['person', 'someone', 'man', 'woman', 'girl', 'boy', 'guy'],
    'picture': ['frame', 'portrait', 'painting', 'photo', 'artwork', 'poster', 'mural'],
    'pillow': ['cushion'],
    'pipe': [],
    'plant': [],
    'printer': [],
    'radiator': ['air conditioning'],
    'rail': ['bar'],
    'recycling bin': [],
    'shoes': ['flip flops'],
    'sign': [],
    'sink': [],
    'shelf': ['shelve'],
    'soap dish': ['soap'],
    'sofa chair': [],
    'stool': [],
    'storage bin': [],
    'suitcase': [],
    'table': ['stand'],
    'telephone': ['phone'],
    'toilet': [],
    'toilet paper': ['paper'],
    'towel': [],
    'trash can': ['trashcan', 'can'],
    'wardrobe closet': ['wardrobe', 'closet', 'locker'],
    'whiteboard': [],
    'window': []
})

instance_to_group = defaultdict(None)
instance_to_group.update({
    'armchair': 'chair',
    'backpack': 'bag',
    'bag': 'bag',
    'bar': None,
    'bathroom stall': None,
    'bathroom stall door': 'door',
    'bed': 'bed',
    'bench': 'chair',
    'blackboard': 'board',
    'blanket': None,
    'board': 'board',
    'book': 'book',
    'books': 'book',
    'bookshelf': 'shelf',
    'bottle': None,
    'box': None,
    'cabinet': 'cabinet',
    'cabinets': 'cabinet',
    'cart': None,
    'chair': 'chair',
    'clothes': 'clothes',
    'clothing': 'clothes',
    'coffee maker': None,
    'coffee table': 'table',
    'computer tower': 'computer',
    'couch': 'chair',
    'cup': None,
    'curtain': 'curtain',
    'desk': 'table',
    'door': 'door',
    'doors': 'door',
    'dresser': None,
    'end table': 'table',
    'file cabinet': 'cabinet',
    'keyboard': None,
    'kitchen cabinet': 'cabinet',
    'kitchen cabinets': 'cabinet',
    'kitchen counter': 'counter',
    'lamp': 'lamp',
    'laptop': 'computer',
    'laundry hamper': 'bin',
    'light': 'lamp',
    'microwave': None,
    'mirror': None,
    'monitor': None,
    'mouse': None,
    'nightstand': None,
    'office chair': 'chair',
    'ottoman': 'chair',
    'oven': 'stove',
    'paper towel dispenser': 'dispenser',
    'person': None,
    'picture': None,
    'pillow': None,
    'pipe': None,
    'plant': 'pot',
    'printer': 'copier',
    'radiator': None,
    'rail': None,
    'recycling bin': 'trash can',
    'shelf': 'shelf',
    'shoes': None,
    'sign': None,
    'sink': None,
    'soap dish': None,
    'sofa chair': 'chair',
    'stool': 'chair',
    'storage bin': 'bin',
    'suitcase': 'bag',
    'table': 'table',
    'telephone': None,
    'toilet': None,
    'toilet paper': 'paper',
    'towel': None,
    'trash can': 'trash can',
    'wardrobe closet': 'closet',
    'whiteboard': 'board',
    'window': None})

group_members = defaultdict(list)
group_members.update(
    {'bag': ['backpack', 'bag of coffee beans', 'duffel bag', 'garbage bag', 'grocery bag', 'golf bag', 'food bag',
             'ikea bag', 'laundry bag', 'messenger bag', 'paper bag', 'shopping bag', 'trash bag', 'suitcase'],
     'bed': ['mattress'],
     'bin': ['storage bin', 'laundry hamper'],
     'board': ['boards', 'whiteboard', 'bulletin board'],
     'book': ['books', 'music book'],
     'cabinet': ['open kitchen cabinet', 'trash cabinet', 'kitchen cabinet', 'file cabinet', 'wardrobe cabinet',
                 'bathroom cabinet', 'bathroom vanity', 'file cabinets', 'kitchen cabinets', 'cabinets'],
     'chair': ['office chair', 'armchair', 'sofa chair', 'massage chair', 'recliner chair', 'rocking chair', 'stool',
               'couch', 'ottoman'],
     'closet': ['wardrobe closet', 'wardrobe'],
     'clothes': ['cloth', 'sock', 'kitchen apron', 'costume'],
     'computer': ['laptop', 'computer tower'],
     'copier': ['printer'],
     'counter': ['bathroom counter', 'kitchen counter'],
     'curtain': ['blinds', 'curtains'],
     'door': ['shower doors', 'glass doors', 'mirror doors', 'garage door', 'doors', 'doorframe', 'closet doorframe',
              'bathroom stall door', 'shower door', 'sliding wood door', 'closet door', 'cabinet doors', 'closet doors',
              'cabinet door'],
     'hamper': ['laundry hamper'],
     'lamp': ['lamp base', 'desk lamp', 'wall lamp', 'table lamp', 'ceiling lamp', 'night lamp', 'light'],
     'paper': ['toilet paper'],
     'shelf': ['organizer shelf', 'pantry shelf', 'bookshelves', 'closet shelf', 'storage shelf', 'bookshelf'],
     'stove': ['oven'],
     'table': ['coffee table', 'dining table', 'folded table', 'round table', 'side table', 'air hockey table',
               'end table', 'desk'],
     'trash can': ['recycling bin', 'trash bin', 'trash bag']})

# Generated using Pattern package, there may non correct plurals nor singular nouns but it is ok
to_plural = {'armchair': 'armchairs', 'backpack': 'backpacks', 'bag': 'bags', 'bar': 'bars',
             'bathroom stall': 'bathroom stalls', 'bathroom stall door': 'bathroom stall doors', 'bed': 'beds',
             'bench': 'benches', 'blackboard': 'blackboards', 'blanket': 'blankets', 'board': 'boards', 'book': 'books',
             'books': 'bookss', 'bookshelf': 'bookshelves', 'bottle': 'bottles', 'box': 'boxes', 'cabinet': 'cabinets',
             'cabinets': 'cabinetss', 'cart': 'carts', 'chair': 'chairs', 'clothes': 'clothess',
             'clothing': 'clothings', 'coffee maker': 'coffee makers', 'coffee table': 'coffee tables',
             'computer tower': 'computer towers', 'couch': 'couches', 'cup': 'cups', 'curtain': 'curtains',
             'desk': 'desks', 'door': 'doors', 'doors': 'doorss', 'dresser': 'dressers', 'end table': 'end tables',
             'file cabinet': 'file cabinets', 'keyboard': 'keyboards', 'kitchen cabinet': 'kitchen cabinets',
             'kitchen cabinets': 'kitchen cabinetss', 'kitchen counter': 'kitchen counters', 'lamp': 'lamps',
             'laptop': 'laptops', 'laundry hamper': 'laundry hampers', 'light': 'lights', 'microwave': 'microwaves',
             'mirror': 'mirrors', 'monitor': 'monitors', 'mouse': 'mice', 'nightstand': 'nightstands',
             'office chair': 'office chairs', 'ottoman': 'ottomen', 'oven': 'ovens',
             'paper towel dispenser': 'paper towel dispensers', 'person': 'people', 'picture': 'pictures',
             'pillow': 'pillows', 'pipe': 'pipes', 'plant': 'plants', 'printer': 'printers', 'radiator': 'radiators',
             'rail': 'rails', 'recycling bin': 'recycling bins', 'shelf': 'shelves', 'shoes': 'shoess', 'sign': 'signs',
             'sink': 'sinks', 'soap dish': 'soap dishes', 'sofa chair': 'sofa chairs', 'stool': 'stools',
             'storage bin': 'storage bins', 'suitcase': 'suitcases', 'table': 'tables', 'telephone': 'telephones',
             'toilet': 'toilets', 'toilet paper': 'toilet papers', 'towel': 'towels', 'trash can': 'trash cans',
             'wardrobe closet': 'wardrobe closets', 'whiteboard': 'whiteboards', 'window': 'windows'}

to_singular = {'books': 'book', 'cabinets': 'cabinet', 'clothes': 'clothe', 'doors': 'door',
               'kitchen cabinets': 'kitchen cabinet', 'shoes': 'shoe'}
