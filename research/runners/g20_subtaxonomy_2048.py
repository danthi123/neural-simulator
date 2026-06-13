"""G.20 2048-concept WITHIN-CLUSTER sub-taxonomy (frozen authored artifact).

The production cortex shards 2048 concepts into 32 semantic clusters of 64
(g20_vocab_spec_2048.ALL_CLUSTERS_2048). For the learned-graded cortex to
generalize MEANINGFULLY *within* a bridge ("a cat is like a dog because both
are pets"), each cluster's 64 words must carry a real sub-category structure:
8 within-cluster SEMANTIC sub-groups of 8. Those 8 sub-groups become the
ground-truth similarity blocks (S_true) the brain-based learn reproduces, so
the cortex's "cat~dog" generalization is semantically meaningful rather than
arbitrary.

This module is a STATIC AUTHORED ARTIFACT: pure Python data + assertions. No
GPU, no ML, no network, no edits to sim/. It mirrors g20_vocab_spec_2048's
additive + import-time-assert style.

  SUBTAXONOMY_2048: dict[str, dict[str, list[str]]]
      cluster-name -> {sub-group-name -> 8-word list}, all 32 clusters in the
      SAME ORDER as ALL_CLUSTERS_2048, each an EXACT partition of that
      cluster's 64 words (every word used once, no extras, no drops).

  cluster_sublabels(cluster_name) -> (words, sublabels)
      the cluster's 64 words in a fixed order (concatenation of the 8
      sub-groups in declaration order) + each word's sub-group id (0..7).
      A corpus generator / per-bridge S_true consumes this: the sub-group id
      IS the within-cluster similarity block.

The import-time EXACT-PARTITION assertion is the correctness net: a curation
slip (a word in two sub-groups, a missing word, an extra) fails at IMPORT and
in the test, never silently. Iterate by running
`python -c "import research.runners.g20_subtaxonomy_2048"` until clean.

The downstream gate re-checks each sub-group's coherence; a few clusters
(function words: abstract_relations, question_discourse, quantity_number_words,
time_words, spatial_words) sub-group by sub-FUNCTION rather than crisp taxonomy
-- flagged in the module docstring and the controller report. Every partition
is still EXACT (the assert enforces it).
"""
from __future__ import annotations

from research.runners.g20_vocab_spec_2048 import ALL_CLUSTERS_2048

# ---------------------------------------------------------------------------
# 32 clusters x 8 sub-groups x 8 words. Same cluster order as the vocab spec.
# Each inner dict is an EXACT partition of ALL_CLUSTERS_2048[cluster].
# ---------------------------------------------------------------------------

SUBTAXONOMY_2048: dict[str, dict[str, list[str]]] = {
    # === mammals ===========================================================
    "mammals": {
        "pets_domestic": ["dog", "cat", "kitten", "puppy", "hamster",
                          "ferret", "rabbit", "mole"],
        "farm_hooved": ["horse", "cow", "pig", "sheep", "goat", "donkey",
                        "mule", "lamb"],
        "big_cats": ["lion", "tiger", "leopard", "cheetah", "panther",
                     "lynx", "jaguar", "hyena"],
        "large_herbivores": ["elephant", "zebra", "giraffe", "rhino",
                             "hippo", "camel", "llama", "foal"],
        "rodents_small": ["mouse", "rat", "squirrel", "beaver", "hedgehog",
                          "porcupine", "bat", "bull"],
        "primates": ["monkey", "ape", "gorilla", "panda", "koala",
                     "kangaroo", "yak", "boar"],
        "marine_mammals": ["whale", "dolphin", "seal", "otter", "moose",
                           "antelope", "bison", "buffalo"],
        "wild_carnivores": ["fox", "wolf", "bear", "deer", "badger",
                            "skunk", "raccoon", "weasel"],
    },
    # === birds =============================================================
    "birds": {
        "waterfowl": ["duck", "goose", "swan", "gull", "tern", "heron",
                      "stork", "crane"],
        "poultry_gamefowl": ["chicken", "hen", "rooster", "turkey", "quail",
                             "pheasant", "partridge", "peacock"],
        "raptors": ["eagle", "hawk", "owl", "falcon", "vulture", "kestrel",
                    "buzzard", "albatross"],
        "corvids": ["crow", "raven", "magpie", "jay", "blackbird", "grackle",
                    "cardinal", "oriole"],
        "songbirds": ["robin", "sparrow", "finch", "wren", "swallow",
                      "starling", "lark", "thrush"],
        "small_perching": ["warbler", "canary", "cuckoo", "nightingale",
                           "vireo", "junco", "pigeon", "dove"],
        "tropical_woodland": ["parrot", "parakeet", "woodpecker",
                              "kingfisher", "hummingbird", "toucan",
                              "flamingo", "pelican"],
        "flightless_seabirds": ["penguin", "ostrich", "emu", "puffin",
                                "cormorant", "seagull", "moorhen", "bird"],
    },
    # === fish_reptiles =====================================================
    "fish_reptiles": {
        "freshwater_fish": ["fish", "salmon", "trout", "bass", "carp",
                            "perch", "pike", "minnow"],
        "saltwater_fish": ["tuna", "cod", "herring", "mackerel", "sardine",
                           "anchovy", "haddock", "pollock"],
        "big_predator_fish": ["shark", "swordfish", "marlin", "barracuda",
                              "piranha", "sturgeon", "grouper", "snapper"],
        "small_oddfish": ["eel", "ray", "catfish", "guppy", "goldfish",
                          "stingray", "moray", "mullet"],
        "flatfish_smallsnakes": ["flounder", "halibut", "tilapia", "adder",
                                 "asp", "garter", "viper", "boa"],
        "amphibians": ["frog", "toad", "newt", "salamander", "tadpole",
                       "terrapin", "turtle", "tortoise"],
        "lizards": ["lizard", "gecko", "iguana", "chameleon", "skink",
                    "monitor", "komodo", "constrictor"],
        "snakes_crocs": ["snake", "cobra", "python", "rattlesnake", "mamba",
                         "anaconda", "crocodile", "alligator"],
    },
    # === insects ===========================================================
    "insects": {
        "bees_wasps": ["bee", "wasp", "hornet", "horntail", "bumblebee",
                       "sawfly", "botfly", "antlion"],
        "biting_flies": ["mosquito", "gnat", "midge", "horsefly", "blowfly",
                         "mayfly", "gadfly", "lacewing"],
        "beetles": ["beetle", "ladybug", "weevil", "firefly", "cicada",
                    "katydid", "earwig", "stinkbug"],
        "roaches_crickets": ["cockroach", "termite", "cricket", "grasshopper",
                             "locust", "mantis", "roach", "hopper"],
        "moths_larvae": ["moth", "butterfly", "caterpillar", "larva", "grub",
                         "maggot", "pupa", "silkworm"],
        "dragonflies_tiny": ["dragonfly", "damselfly", "aphid", "thrips",
                             "springtail", "chigger", "nit", "earworm"],
        "parasites": ["flea", "louse", "tick", "mite", "bedbug", "weaver",
                      "harvestman", "worm"],
        "arachnids_crawlers": ["spider", "scorpion", "centipede", "millipede",
                               "slug", "snail", "tarantula", "ant"],
    },
    # === fruits ============================================================
    "fruits": {
        "tree_orchard": ["apple", "pear", "peach", "plum", "cherry",
                         "apricot", "nectarine", "quince"],
        "citrus": ["orange", "lemon", "lime", "grapefruit", "tangerine",
                   "clementine", "mandarin", "kumquat"],
        "berries_common": ["berry", "strawberry", "raspberry", "blueberry",
                           "blackberry", "cranberry", "currant", "gooseberry"],
        "berries_exotic": ["mulberry", "elderberry", "honeyberry",
                           "boysenberry", "loganberry", "huckleberry",
                           "lingonberry", "cloudberry"],
        "tropical": ["banana", "mango", "papaya", "pineapple", "kiwi",
                     "coconut", "guava", "lychee"],
        "melons": ["melon", "watermelon", "cantaloupe", "honeydew",
                   "grape", "fig", "date", "olive"],
        "exotic_tropical2": ["pomegranate", "persimmon", "passion",
                             "dragonfruit", "starfruit", "jackfruit",
                             "durian", "soursop"],
        "savory_misc": ["fruit", "avocado", "tomato", "plantain", "tamarind",
                        "salmonberry", "damson", "sloe"],
    },
    # === vegetables ========================================================
    "vegetables": {
        "roots_tubers": ["carrot", "potato", "radish", "turnip", "beet",
                         "parsnip", "rutabaga", "swede"],
        "alliums": ["onion", "garlic", "leek", "shallot", "chive",
                    "scallion", "fennel", "horseradish"],
        "leafy_greens": ["lettuce", "cabbage", "spinach", "kale", "chard",
                         "arugula", "watercress", "endive"],
        "brassicas_stems": ["broccoli", "cauliflower", "celery", "asparagus",
                            "artichoke", "kohlrabi", "rhubarb", "samphire"],
        "squashes": ["cucumber", "pumpkin", "squash", "zucchini", "gourd",
                     "marrow", "courgette", "celeriac"],
        "nightshade_pods": ["pepper", "eggplant", "okra", "aubergine",
                            "pea", "bean", "corn", "sweetcorn"],
        "legumes": ["chickpea", "lentil", "soybean", "edamame", "sprout",
                    "cress", "collard", "mustard"],
        "exotic_roots": ["mushroom", "yam", "cassava", "ginger", "salsify",
                         "chicory", "beetroot", "mangetout"],
    },
    # === prepared_foods ====================================================
    "prepared_foods": {
        "breads_baked": ["bread", "toast", "bagel", "pretzel", "cracker",
                         "wafer", "scone", "crumb"],
        "main_dishes": ["sandwich", "pizza", "burger", "taco", "burrito",
                        "quiche", "casserole", "fritter"],
        "pasta_grains": ["pasta", "noodle", "rice", "cereal", "oatmeal",
                         "porridge", "dumpling", "stuffing"],
        "soups_stews": ["soup", "stew", "salad", "curry", "chili", "gravy",
                        "sauce", "dip"],
        "desserts_baked": ["pie", "cake", "cookie", "biscuit", "muffin",
                           "pancake", "waffle", "donut"],
        "sweets": ["pastry", "pudding", "custard", "jelly", "jam", "candy",
                   "chocolate", "honey"],
        "dairy_eggs": ["butter", "cheese", "yogurt", "cream", "egg", "omelet",
                       "spread", "topping"],
        "meats_dough": ["bacon", "sausage", "ham", "steak", "roast", "dough",
                        "crust", "filling"],
    },
    # === drinks ============================================================
    "drinks": {
        "nonalcoholic_cold": ["water", "milk", "juice", "soda", "lemonade",
                              "cola", "tonic", "punch"],
        "hot_drinks": ["tea", "coffee", "cocoa", "espresso", "latte",
                       "cappuccino", "mocha", "decaf"],
        "beer_cider": ["cider", "beer", "ale", "malt", "lager", "stout",
                       "brew", "brewage"],
        "wine_fortified": ["wine", "champagne", "sherry", "vermouth", "mead",
                           "cordial", "potion", "elixir"],
        "spirits": ["whiskey", "vodka", "rum", "gin", "brandy", "liquor",
                    "cocktail", "sake"],
        "blended_frozen": ["smoothie", "milkshake", "frappe", "slush",
                           "sherbet", "spritzer", "fizz", "infusion"],
        "fermented_dairy": ["kombucha", "kefir", "buttermilk", "eggnog",
                            "broth", "nectar", "syrup", "barley"],
        "generic_actions": ["sip", "gulp", "quencher", "beverage", "drip",
                            "draught", "swill", "grog"],
    },
    # === land_vehicles =====================================================
    "land_vehicles": {
        "cars": ["car", "sedan", "coupe", "convertible", "hatchback",
                 "roadster", "racecar", "dragster"],
        "trucks_utility": ["truck", "van", "pickup", "lorry", "hauler",
                           "dumper", "minivan", "rover"],
        "buses_taxis": ["bus", "taxi", "coach", "minibus", "limousine",
                        "jeep", "wagon", "buggy"],
        "two_wheelers": ["bike", "bicycle", "motorcycle", "scooter", "moped",
                         "unicycle", "tricycle", "tandem"],
        "rail": ["train", "tram", "subway", "trolley", "monorail",
                 "locomotive", "boxcar", "caboose"],
        "emergency_construction": ["ambulance", "firetruck", "forklift",
                                   "bulldozer", "excavator", "tractor",
                                   "trailer", "snowplow"],
        "human_powered_carts": ["cart", "handcart", "wheelbarrow", "pram",
                                "stroller", "trundle", "rickshaw", "chariot"],
        "snow_misc": ["sled", "sleigh", "skateboard", "rollerblade",
                      "snowmobile", "segway", "carriage", "hearse"],
    },
    # === air_water_vehicles ================================================
    "air_water_vehicles": {
        "powered_aircraft": ["plane", "jet", "airplane", "biplane",
                             "seaplane", "airliner", "fighter", "bomber"],
        "rotorcraft_lighter": ["helicopter", "glider", "blimp", "balloon",
                              "airship", "zeppelin", "parachute", "drone"],
        "spacecraft": ["rocket", "shuttle", "spacecraft", "satellite",
                       "hovercraft", "hydrofoil", "pontoon", "outrigger"],
        "small_boats": ["boat", "canoe", "kayak", "raft", "dinghy", "rowboat",
                        "skiff", "punt"],
        "sailing": ["yacht", "sailboat", "schooner", "sloop", "ketch",
                    "cutter", "junk", "dhow"],
        "cargo_ships": ["ship", "barge", "tanker", "freighter", "liner",
                        "steamer", "scow", "wherry"],
        "warships": ["submarine", "frigate", "destroyer", "cruiser",
                     "battleship", "carrier", "corvette", "galleon"],
        "ferries_misc": ["ferry", "tugboat", "trawler", "gondola",
                         "houseboat", "catamaran", "speedboat", "lifeboat"],
    },
    # === hand_tools ========================================================
    "hand_tools": {
        "striking_cutting": ["hammer", "mallet", "axe", "hatchet", "pickaxe",
                            "chisel", "crowbar", "gouge"],
        "fasten_drive": ["screwdriver", "wrench", "pliers", "drill", "clamp",
                         "vice", "stapler", "awl"],
        "saws_blades": ["saw", "knife", "blade", "scissors", "shears",
                        "clippers", "jigsaw", "bradawl"],
        "kitchen_tools": ["spoon", "fork", "ladle", "spatula", "whisk",
                          "peeler", "grater", "tongs"],
        "garden_dig": ["rake", "hoe", "shovel", "spade", "trowel", "broom",
                       "mop", "bucket"],
        "measuring_marking": ["ruler", "tape", "level", "compass", "pencil",
                             "pen", "key", "file"],
        "finishing": ["brush", "scraper", "sander", "rasp", "sieve",
                      "strainer", "hook", "anchor"],
        "fixings": ["needle", "pin", "nail", "screw", "bolt", "nut", "shim",
                    "rivet"],
    },
    # === machines ==========================================================
    "machines": {
        "power_plant": ["engine", "motor", "pump", "generator", "turbine",
                        "compressor", "dynamo", "alternator"],
        "computing": ["computer", "laptop", "tablet", "phone", "printer",
                      "scanner", "copier", "machine"],
        "media_devices": ["camera", "radio", "television", "speaker",
                          "console", "projector", "amplifier", "transformer"],
        "peripherals": ["keyboard", "trackpad", "router", "modem", "battery",
                        "charger", "switch", "circuit"],
        "climate_appliances": ["fan", "heater", "furnace", "boiler",
                              "refrigerator", "freezer", "thermostat",
                              "windmill"],
        "kitchen_appliances": ["oven", "stove", "microwave", "toaster",
                              "blender", "mixer", "crusher", "loom"],
        "cleaning_lifting": ["dishwasher", "washer", "dryer", "vacuum",
                            "robot", "elevator", "escalator", "conveyor"],
        "mechanical_parts": ["winch", "hoist", "lathe", "spindle", "gear",
                            "piston", "valve", "sensor"],
    },
    # === clothing ==========================================================
    "clothing": {
        "tops": ["shirt", "blouse", "sweater", "hoodie", "cardigan", "vest",
                 "tie", "scarf"],
        "bottoms": ["pants", "trousers", "jeans", "shorts", "skirt",
                    "leggings", "tights", "overalls"],
        "outerwear": ["jacket", "coat", "blazer", "suit", "raincoat",
                      "poncho", "cloak", "shawl"],
        "headwear": ["hat", "cap", "beanie", "bonnet", "helmet", "hood",
                     "collar", "cuff"],
        "footwear": ["shoe", "boot", "sandal", "slipper", "sneaker", "loafer",
                     "sock", "stocking"],
        "dresses_sleepwear": ["dress", "robe", "gown", "pajamas", "nightgown",
                             "swimsuit", "bikini", "apron"],
        "fasteners_parts": ["belt", "buckle", "button", "zipper", "lace",
                            "sleeve", "hem", "pocket"],
        "undergarments_material": ["glove", "mitten", "underwear", "bra",
                                   "uniform", "garment", "fabric", "cloth"],
    },
    # === furniture =========================================================
    "furniture": {
        "seating": ["chair", "stool", "bench", "couch", "sofa", "armchair",
                    "recliner", "ottoman"],
        "tables_desks": ["table", "desk", "easel", "rack", "hanger", "hamper",
                         "frame", "mirror"],
        "storage": ["shelf", "bookcase", "cabinet", "cupboard", "drawer",
                    "dresser", "wardrobe", "closet"],
        "beds_bedding": ["bed", "bunk", "crib", "cradle", "mattress", "pillow",
                         "blanket", "quilt"],
        "soft_furnishings": ["cushion", "rug", "carpet", "mat", "nightstand",
                            "curtain", "blind", "shutter"],
        "lighting_decor": ["lamp", "chandelier", "clock", "vase", "basket",
                           "book", "ball", "bag"],
        "kitchenware": ["bowl", "jar", "pot", "pan", "kettle", "tray", "cup",
                        "plate"],
        "tableware": ["mug", "glass", "bottle", "dish", "saucer", "napkin",
                      "tablecloth", "box"],
    },
    # === buildings =========================================================
    "buildings": {
        "homes": ["house", "home", "cabin", "cottage", "mansion", "castle",
                  "palace", "tower"],
        "education_culture": ["school", "library", "museum", "theater",
                             "gym", "stadium", "arena", "building"],
        "retail_commerce": ["shop", "store", "market", "mall", "bakery",
                            "cafe", "restaurant", "bank"],
        "worship": ["church", "temple", "mosque", "cathedral", "chapel",
                    "prison", "jail", "hotel"],
        "work_industry": ["office", "factory", "warehouse", "barn", "shed",
                          "garage", "motel", "inn"],
        "health_transit": ["hospital", "clinic", "station", "airport",
                          "harbor", "port", "bridge", "tunnel"],
        "openings_barriers": ["window", "door", "gate", "fence", "wall",
                             "roof", "garden", "park"],
        "interior_spaces": ["floor", "ceiling", "stairs", "hallway", "room",
                           "basement", "attic", "road"],
    },
    # === body_parts ========================================================
    "body_parts": {
        "face": ["face", "cheek", "chin", "jaw", "lip", "forehead",
                 "eyebrow", "scalp"],
        "head_sensory": ["head", "eye", "ear", "nose", "mouth", "tongue",
                         "eyelash", "nostril"],
        "mouth_teeth": ["tooth", "gum", "throat", "neck", "hair", "skin",
                        "skull", "brain"],
        "arms": ["arm", "shoulder", "elbow", "wrist", "hand", "palm",
                 "knuckle", "cuticle"],
        "fingers": ["finger", "thumb", "leg", "thigh", "shin", "calf",
                    "limb", "torso"],
        "legs_feet": ["foot", "knee", "ankle", "heel", "toe", "hip", "rump",
                      "tendon"],
        "torso": ["chest", "rib", "belly", "stomach", "waist", "navel",
                  "spine", "muscle"],
        "internal_organs": ["heart", "lung", "liver", "kidney", "nerve",
                            "vein", "blood", "bone"],
    },
    # === plants_trees ======================================================
    "plants_trees": {
        "deciduous_trees": ["tree", "oak", "maple", "birch", "elm", "willow",
                            "ash", "beech"],
        "conifers_giants": ["pine", "cedar", "spruce", "fir", "poplar",
                           "sequoia", "bamboo", "cactus"],
        "flowers_garden": ["flower", "rose", "tulip", "daisy", "lily", "iris",
                           "orchid", "sunflower"],
        "flowers_wild": ["poppy", "daffodil", "geranium", "jasmine",
                         "marigold", "petunia", "clover", "thistle"],
        "low_plants": ["bush", "shrub", "vine", "weed", "fern", "moss", "ivy",
                       "nettle"],
        "wood_parts": ["bark", "twig", "branch", "trunk", "bough", "stump",
                       "stem", "shoot"],
        "growth_parts": ["leaf", "petal", "bud", "blossom", "sapling",
                         "seedling", "seed", "root"],
        "ground_cover": ["grass", "reed", "foliage", "thorn", "hedge",
                         "thicket", "grove", "lichen"],
    },
    # === weather_nature ====================================================
    "weather_nature": {
        "precipitation": ["rain", "snow", "hail", "drizzle", "shower",
                          "downpour", "sleet", "dew"],
        "atmospheric": ["wind", "cloud", "fog", "mist", "frost", "breeze",
                        "gust", "gale"],
        "storms": ["storm", "thunder", "lightning", "hurricane", "tornado",
                   "blizzard", "earthquake", "flood"],
        "celestial": ["sun", "moon", "sky", "star", "planet", "comet",
                      "meteor", "rainbow"],
        "fire_light": ["fire", "ice", "sunshine", "shadow", "sunset",
                       "sunrise", "drought", "humidity"],
        "day_cycle": ["dawn", "dusk", "twilight", "heat", "chill", "warmth",
                      "weather", "climate"],
        "seasons": ["season", "spring", "summer", "autumn", "winter", "wave",
                    "ocean", "sea"],
        "landforms": ["mountain", "valley", "hill", "cliff", "river",
                      "desert", "forest", "lake"],
    },
    # === kinship_people ====================================================
    "kinship_people": {
        "nuclear_family": ["mother", "father", "parent", "son", "daughter",
                          "brother", "sister", "sibling"],
        "extended_family": ["grandfather", "grandmother", "grandparent",
                           "grandson", "granddaughter", "uncle", "aunt",
                           "cousin"],
        "in_laws_relatives": ["nephew", "niece", "husband", "wife", "spouse",
                             "family", "relative", "kin"],
        "lineage_partners": ["ancestor", "descendant", "twin", "partner",
                            "widow", "orphan", "guardian", "nanny"],
        "children": ["baby", "child", "toddler", "infant", "kid", "boy",
                     "girl", "person"],
        "adults_social": ["man", "woman", "adult", "elder", "friend",
                          "neighbor", "stranger", "guest"],
        "groups_host": ["host", "people", "crowd", "comrade", "citizen",
                        "leader", "worker", "farmer"],
        "occupations_royalty": ["teacher", "student", "doctor", "nurse",
                              "king", "queen", "prince", "princess"],
    },
    # === motion_verbs ======================================================
    "motion_verbs": {
        "running_fast": ["run", "dash", "sprint", "jog", "race", "chase",
                         "scamper", "dart"],
        "walking": ["go", "come", "walk", "march", "stride", "stroll", "pace",
                    "wander"],
        "jumping": ["jump", "hop", "leap", "bounce", "skip", "lunge", "swing",
                    "sway"],
        "flying_air": ["fly", "soar", "swoop", "glide", "drift", "float",
                       "rise", "dive"],
        "falling_down": ["fall", "sink", "plunge", "slide", "roll", "trip",
                         "stumble", "stagger"],
        "rotating": ["turn", "spin", "twirl", "rock", "shake", "tremble",
                     "wobble", "climb"],
        "body_locomotion": ["crawl", "ride", "swim", "sit", "stand", "move",
                            "tiptoe", "sneak"],
        "object_actions": ["throw", "catch", "kick", "hit", "stop", "flee",
                          "escape", "roam"],
    },
    # === perception_verbs ==================================================
    "perception_verbs": {
        "see_general": ["look", "see", "watch", "view", "observe", "notice",
                        "witness", "behold"],
        "look_intently": ["stare", "gaze", "peer", "scrutinize", "examine",
                         "inspect", "study", "survey"],
        "quick_glances": ["glance", "peek", "spot", "glimpse", "blink",
                          "wink", "squint", "ogle"],
        "find_detect": ["find", "detect", "discover", "recognize", "identify",
                        "sense", "perceive", "discern"],
        "touch_contact": ["touch", "stroke", "rub", "pat", "poke", "prod",
                          "grope", "fondle"],
        "handle_grasp": ["caress", "tickle", "graze", "skim", "handle",
                         "grasp", "clutch", "leer"],
        "smell_taste": ["smell", "taste", "feel", "sniff", "savor", "lick",
                        "sight", "regard"],
        "search_monitor": ["scan", "search", "seek", "spy", "supervise",
                          "overhear", "hear", "listen"],
    },
    # === communication_verbs ===============================================
    "communication_verbs": {
        "speak_general": ["speak", "talk", "chat", "converse", "discuss",
                          "say", "tell", "mention"],
        "read_write": ["read", "write", "spell", "quote", "recite", "narrate",
                       "translate", "describe"],
        "ask_answer": ["ask", "answer", "reply", "respond", "call", "request",
                       "beg", "plead"],
        "declare_state": ["state", "declare", "announce", "report", "remark",
                          "comment", "explain", "claim"],
        "loud_speech": ["shout", "yell", "scream", "cheer", "rant", "boast",
                        "gossip", "lecture"],
        "quiet_speech": ["whisper", "murmur", "mumble", "hum", "sing", "chant",
                         "preach", "thank"],
        "argue_assert": ["argue", "debate", "insist", "deny", "agree",
                         "promise", "swear", "warn"],
        "advise_command": ["advise", "suggest", "command", "order", "greet",
                          "praise", "scold", "blame"],
    },
    # === manipulation_verbs ================================================
    "manipulation_verbs": {
        "push_pull": ["push", "pull", "lift", "lower", "press", "squeeze",
                      "shove", "grab"],
        "open_close": ["open", "close", "give", "take", "hold", "drop",
                       "place", "remove"],
        "cut_break": ["cut", "break", "tear", "rip", "snap", "crack", "split",
                      "slice"],
        "crush_destroy": ["crush", "smash", "grind", "chop", "twist", "bend",
                          "fold", "stir"],
        "build_make": ["build", "make", "fix", "carry", "bring", "send",
                       "stack", "pile"],
        "cook_food": ["eat", "drink", "cook", "mix", "pour", "fill", "drain",
                      "sort"],
        "clean_wipe": ["wash", "wipe", "scrub", "rinse", "dab", "sweep",
                       "polish", "paint"],
        "join_attach": ["glue", "stick", "bind", "untie", "knot", "wrap",
                        "pack", "arrange"],
    },
    # === emotion_states ====================================================
    "emotion_states": {
        "positive_feelings": ["happy", "joy", "love", "hope", "glad",
                             "cheerful", "delighted", "content"],
        "negative_feelings": ["sad", "grief", "hate", "worry", "gloomy",
                             "moody", "miserable", "depressed"],
        "anger": ["angry", "anger", "furious", "annoyed", "frustrated",
                  "jealous", "guilty", "ashamed"],
        "fear_anxiety": ["scared", "fear", "afraid", "nervous", "anxious",
                         "panic", "shy", "lonely"],
        "energy_states": ["excited", "eager", "brave", "proud", "hopeful",
                          "grateful", "restless", "calm"],
        "tiredness_rest": ["tired", "weary", "bored", "relax", "rest", "yawn",
                           "doze", "sleep"],
        "expressions": ["laugh", "cry", "smile", "frown", "weep", "sob",
                        "sigh", "groan"],
        "activity_states": ["play", "work", "wait", "wake", "rejoice",
                           "mourn", "dream", "fret"],
    },
    # === size_shape_adj ====================================================
    "size_shape_adj": {
        "very_big": ["big", "large", "huge", "giant", "massive", "enormous",
                     "vast", "gigantic"],
        "extreme_big": ["colossal", "broad", "wide", "bulky", "chunky",
                        "plump", "squat", "lanky"],
        "small": ["small", "little", "tiny", "teeny", "miniature", "petite",
                  "slim", "slender"],
        "length_height": ["tall", "short", "long", "high", "low", "steep",
                          "shallow", "wiry"],
        "thickness_depth": ["thin", "thick", "deep", "narrow", "compact",
                            "solid", "hollow", "tapered"],
        "round_shapes": ["round", "circular", "oval", "curved", "spherical",
                         "cylindrical", "domed", "convex"],
        "angular_shapes": ["flat", "square", "straight", "rectangular",
                          "triangular", "angular", "pointed", "concave"],
        "edges_surface": ["sharp", "blunt", "jagged", "smooth", "bumpy",
                          "lumpy", "crooked", "bent"],
    },
    # === color_adj =========================================================
    "color_adj": {
        "reds_pinks": ["red", "crimson", "scarlet", "maroon", "ruby", "blush",
                       "coral", "rosy"],
        "warm_earth": ["ochre", "amber", "gold", "golden", "tan", "beige",
                       "eggshell", "ivory"],
        "greens": ["green", "hunter", "chartreuse", "emerald", "jade", "teal",
                   "turquoise", "khaki"],
        "blues": ["blue", "aqua", "cyan", "navy", "azure", "cobalt", "indigo",
                  "violet"],
        "purples": ["purple", "lavender", "lilac", "magenta", "fuchsia",
                    "burgundy", "mauve", "pink"],
        "browns_metals": ["brown", "bronze", "copper", "rust", "chestnut",
                          "auburn", "silver", "gray"],
        "neutrals": ["white", "black", "charcoal", "ebony", "pale", "pastel",
                     "dim", "yellow"],
        "qualities": ["bright", "dark", "vivid", "colorful", "shiny", "glossy",
                      "matte", "translucent"],
    },
    # === texture_material_adj ==============================================
    "texture_material_adj": {
        "temperature": ["hot", "cold", "warm", "cool", "damp", "wet", "dry",
                        "loud"],
        "speed_age": ["fast", "slow", "new", "old", "young", "quiet", "well",
                      "sick"],
        "cleanliness_fill": ["clean", "dirty", "full", "empty", "heavy",
                           "light", "true", "false"],
        "hardness": ["hard", "soft", "firm", "rigid", "flexible", "stiff",
                     "elastic", "spongy"],
        "taste_value": ["sweet", "sour", "strong", "weak", "rich", "poor",
                        "good", "bad"],
        "character": ["kind", "mean", "nice", "tough", "tender", "brittle",
                      "crisp", "coarse"],
        "surface_feel": ["rough", "fuzzy", "silky", "fluffy", "grainy",
                         "sticky", "slippery", "greasy"],
        "materials": ["oily", "metallic", "wooden", "plastic", "glassy",
                      "rubbery", "leathery", "woolly"],
    },
    # === time_words ========================================================
    "time_words": {
        "sequence_order": ["now", "then", "before", "after", "first", "last",
                          "next", "again"],
        "relative_days": ["today", "yesterday", "tomorrow", "soon", "late",
                          "early", "once", "later"],
        "frequency": ["always", "never", "often", "sometimes", "frequently",
                      "rarely", "seldom", "occasionally"],
        "conjunctive_time": ["until", "since", "during", "whenever", "while",
                            "meanwhile", "afterward", "beforehand"],
        "day_segments": ["morning", "noon", "afternoon", "evening", "night",
                         "midnight", "moment", "instant"],
        "time_units": ["day", "week", "month", "year", "hour", "minute",
                       "second", "henceforth"],
        "temporal_adverbs": ["previously", "currently", "presently",
                            "eventually", "finally", "immediately",
                            "instantly", "shortly"],
        "recency_periodic": ["recently", "lately", "daily", "weekly",
                           "monthly", "yearly", "hourly", "nowadays"],
    },
    # === spatial_words =====================================================
    "spatial_words": {
        "cardinal_directions": ["north", "south", "east", "west", "up", "down",
                              "left", "right"],
        "near_far": ["here", "there", "near", "far", "nearby", "distant",
                     "off", "onto"],
        "containment": ["in", "out", "on", "inside", "outside", "within",
                        "underneath", "overhead"],
        "vertical": ["under", "above", "below", "top", "bottom", "beneath",
                     "atop", "amid"],
        "front_back": ["front", "back", "behind", "beyond", "forward",
                       "backward", "opposite", "adjacent"],
        "lateral": ["side", "middle", "beside", "alongside", "between",
                    "among", "around", "center"],
        "movement_relative": ["through", "across", "along", "toward", "away",
                            "inward", "outward", "sideways"],
        "location_extent": ["edge", "corner", "upward", "downward",
                          "everywhere", "nowhere", "somewhere", "anywhere"],
    },
    # === quantity_number_words =============================================
    "quantity_number_words": {
        "small_numbers": ["one", "two", "three", "four", "five", "six",
                          "seven", "eight"],
        "teen_numbers": ["nine", "ten", "eleven", "twelve", "thirteen",
                         "fourteen", "fifteen", "sixteen"],
        "teen_high_tens": ["seventeen", "eighteen", "nineteen", "twenty",
                          "thirty", "forty", "fifty", "sixty"],
        "large_numbers": ["seventy", "eighty", "ninety", "hundred",
                         "thousand", "million", "billion", "zero"],
        "groups_pairs": ["dozen", "pair", "couple", "both", "single",
                         "double", "triple", "quarter"],
        "fractions_parts": ["half", "third", "whole", "extra", "amount",
                           "number", "total", "count"],
        "quantifiers_many": ["many", "few", "some", "several", "plenty",
                            "much", "more", "most"],
        "quantifiers_all": ["all", "none", "every", "any", "each", "less",
                           "least", "enough"],
    },
    # === question_discourse ================================================
    "question_discourse": {
        "wh_questions": ["what", "where", "when", "who", "why", "how", "which",
                         "whose"],
        "wh_compounds": ["whom", "whatever", "wherever", "whoever", "however",
                         "whichever", "namely", "regardless"],
        "affirm_negate": ["yes", "no", "okay", "yeah", "yep", "nope", "ok",
                          "indeed"],
        "greetings": ["hello", "goodbye", "hey", "hi", "bye", "farewell",
                      "welcome", "greetings"],
        "courtesy": ["please", "thanks", "sorry", "cheers", "congratulations",
                     "pardon", "excuse", "apology"],
        "exclamations": ["oops", "ouch", "wow", "oh", "ah", "huh", "hmm",
                         "uh"],
        "demonstratives": ["this", "that", "these", "those", "it", "righto",
                           "uhh", "regret"],
        "emphasis_hedge": ["certainly", "absolutely", "exactly", "maybe",
                          "perhaps", "alright", "anyway", "anyhow"],
    },
    # === abstract_relations ================================================
    "abstract_relations": {
        "coordinating": ["and", "or", "but", "nor", "yet", "either", "neither",
                         "so"],
        "causal": ["because", "cause", "reason", "therefore", "thus", "hence",
                   "consequently", "result"],
        "conditional": ["if", "unless", "whether", "although", "though",
                        "whereas", "given", "owing"],
        "modal_verbs": ["can", "will", "must", "should", "would", "could",
                        "might", "did"],
        "stative_verbs": ["do", "is", "have", "want", "need", "instead",
                          "besides", "meaning"],
        "intensifiers": ["very", "too", "also", "only", "not", "same", "other",
                         "another"],
        "additive_connectives": ["moreover", "furthermore", "nonetheless",
                               "regarding", "purpose", "effect", "factor",
                               "condition"],
        "relational_nouns": ["difference", "similarity", "relation",
                           "connection", "contrast", "comparison",
                           "consequence", "implication"],
    },
}

# ---------------------------------------------------------------------------
# Import-time invariants (the correctness net). Run AT IMPORT and in the test.
# ---------------------------------------------------------------------------

# Same 32 clusters, in the SAME order as the vocab spec.
assert list(SUBTAXONOMY_2048.keys()) == list(ALL_CLUSTERS_2048.keys()), (
    "Sub-taxonomy cluster set/order must match ALL_CLUSTERS_2048"
)

for _cname, _subgroups in SUBTAXONOMY_2048.items():
    # Exactly 8 sub-groups.
    assert len(_subgroups) == 8, (
        f"{_cname} has {len(_subgroups)} sub-groups, expected 8"
    )
    # Unique sub-group names within the cluster.
    _sgnames = list(_subgroups.keys())
    assert len(_sgnames) == len(set(_sgnames)), (
        f"{_cname} has duplicate sub-group names: "
        f"{sorted(n for n in _sgnames if _sgnames.count(n) > 1)}"
    )
    _flat: list[str] = []
    for _sgname, _words in _subgroups.items():
        # Exactly 8 words per sub-group.
        assert len(_words) == 8, (
            f"{_cname}/{_sgname} has {len(_words)} words, expected 8"
        )
        for _w in _words:
            assert _w and _w == _w.lower(), (
                f"{_cname}/{_sgname} word not lowercase/non-empty: {_w!r}"
            )
        _flat.extend(_words)
    # EXACT partition of the vocab-spec cluster: no dups, same 64-word set.
    assert len(_flat) == len(set(_flat)), (
        f"{_cname} sub-taxonomy has duplicate words: "
        f"{sorted(w for w in _flat if _flat.count(w) > 1)}"
    )
    _spec_set = set(ALL_CLUSTERS_2048[_cname])
    _missing = _spec_set - set(_flat)
    _extra = set(_flat) - _spec_set
    assert not _missing, f"{_cname} sub-taxonomy MISSING: {sorted(_missing)}"
    assert not _extra, f"{_cname} sub-taxonomy has EXTRAS: {sorted(_extra)}"
    assert len(_flat) == 64, (
        f"{_cname} sub-taxonomy has {len(_flat)} words, expected 64"
    )

# Global total = 2048.
_TOTAL = sum(
    len(_words)
    for _subgroups in SUBTAXONOMY_2048.values()
    for _words in _subgroups.values()
)
assert _TOTAL == 2048, f"Total sub-taxonomy words {_TOTAL}, expected 2048"


# ---------------------------------------------------------------------------
# Consumer helper.
# ---------------------------------------------------------------------------

def cluster_sublabels(cluster_name: str) -> tuple[list[str], list[int]]:
    """Return ``(words, sublabels)`` for one cluster.

    ``words`` is the cluster's 64 words in a fixed order: the concatenation
    of its 8 sub-groups in declaration order. ``sublabels[i]`` is the
    within-cluster sub-group id (0..7) of ``words[i]`` -- exactly 8 words per
    id. A corpus generator / per-bridge S_true consumes this: the sub-group id
    is the within-cluster similarity block (members of the same sub-group are
    the ground-truth "more similar" pairs).

    Raises ``KeyError`` if ``cluster_name`` is not one of the 32 clusters.
    """
    subgroups = SUBTAXONOMY_2048[cluster_name]
    words: list[str] = []
    sublabels: list[int] = []
    for sid, sg_words in enumerate(subgroups.values()):
        words.extend(sg_words)
        sublabels.extend([sid] * len(sg_words))
    return words, sublabels


if __name__ == "__main__":
    print(
        f"G.20 2048-concept within-cluster sub-taxonomy: "
        f"{len(SUBTAXONOMY_2048)} clusters x 8 sub-groups x 8 words = "
        f"{_TOTAL} concepts.\n"
    )
    for _cname, _subgroups in SUBTAXONOMY_2048.items():
        print(f"{_cname}  ({sum(len(w) for w in _subgroups.values())} words):")
        for _sgname, _words in _subgroups.items():
            print(f"    {_sgname:>24} ({len(_words)}): {_words}")
        print()
