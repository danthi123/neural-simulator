"""G.20 32-cluster 2048-concept vocab spec (production sharding tier).

32 hand-curated semantic super-clusters of exactly 64 mutually-similar
concepts each = 2048 globally-unique single-word concepts.

Why semantic-cluster sharding (NOT part-of-speech): the dual/CLS
learned-graded cortex scales to 2048 concepts split across 32 spiking
bridges of 64 concepts each, and within-bridge generalization ("a cat is
like a dog") only works when a bridge's 64 concepts genuinely cluster in
concept space. So each bridge must hold a TAXONOMIC cluster (all mammals,
all colors, all motion verbs), not a part-of-speech mix. This is the key
difference from the validated 320 spec (g20_vocab_spec_320.py), which shards
the same surface forms by part-of-speech across 5 bridges.

Composition of the vocabulary:
  - REUSES the 320 validated base words (g20_vocab_spec_320.ALL_WORDS_64),
    RE-CLUSTERED from their 5 part-of-speech lists into the 32 semantic
    clusters by taxonomy (e.g. "dog" -> mammals, "apple" -> fruits, "run"
    -> motion_verbs, "red" -> color_adj, "inside" -> spatial_words,
    "because" -> abstract_relations). All 320 have a semantic home.
  - ADDS ~1728 new curated words to fill every cluster to exactly 64.

The module-level GLOBAL-UNIQUENESS assert is the correctness net: a
hand-curation collision (the same word landing in two clusters) fails at
IMPORT and in the test, never silently. Iterate by running
`python -c "import research.runners.g20_vocab_spec_2048"` until clean.

This module is ADDITIVE / pure-data: no GPU, no ML, no edits to sim/, no
network. The downstream gate re-checks each cluster's coherence and any
mis-curated member is fixable later.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 32 semantic super-clusters, 64 mutually-similar concepts each.
# Base-320 words (re-clustered by taxonomy) are flagged inline as # base.
# ---------------------------------------------------------------------------

CLUSTER_MAMMALS = [
    # base-320: dog cat mouse horse cow pig sheep bear wolf
    "dog", "cat", "mouse", "horse", "cow", "pig", "sheep", "bear", "wolf",
    "fox", "deer", "rabbit", "goat", "lion", "tiger", "elephant", "monkey",
    "ape", "gorilla", "zebra", "giraffe", "rhino", "hippo", "camel",
    "donkey", "mule", "bull", "yak", "lamb", "foal", "kitten", "puppy",
    "rat", "hamster", "squirrel", "beaver", "otter", "seal", "whale",
    "dolphin", "bat", "mole", "hedgehog", "badger", "skunk", "raccoon",
    "weasel", "ferret", "panda", "koala", "kangaroo", "leopard", "cheetah",
    "panther", "lynx", "bison", "buffalo", "moose", "antelope", "boar",
    "hyena", "jaguar", "llama", "porcupine",
]

CLUSTER_BIRDS = [
    # base-320: bird duck
    "bird", "duck", "goose", "swan", "chicken", "hen", "rooster", "turkey",
    "eagle", "hawk", "owl", "falcon", "vulture", "crow", "raven", "magpie",
    "robin", "sparrow", "finch", "wren", "swallow", "starling", "pigeon",
    "dove", "gull", "tern", "heron", "stork", "crane", "flamingo", "pelican",
    "penguin", "ostrich", "emu", "peacock", "parrot", "parakeet", "canary",
    "cuckoo", "woodpecker", "kingfisher", "hummingbird", "lark", "thrush",
    "blackbird", "jay", "cardinal", "warbler", "oriole", "quail", "pheasant",
    "partridge", "puffin", "albatross", "cormorant", "kestrel", "buzzard",
    "nightingale", "seagull", "moorhen", "toucan", "vireo", "grackle", "junco",
]

CLUSTER_FISH_REPTILES = [
    # base-320: fish frog snake
    "fish", "frog", "snake", "salmon", "trout", "tuna", "cod", "bass",
    "carp", "perch", "pike", "eel", "shark", "ray", "minnow", "catfish",
    "herring", "mackerel", "sardine", "anchovy", "guppy", "goldfish",
    "toad", "newt", "salamander", "lizard", "gecko", "iguana", "chameleon",
    "skink", "turtle", "tortoise", "crocodile", "alligator", "cobra",
    "python", "viper", "adder", "boa", "rattlesnake", "mamba", "anaconda",
    "flounder", "halibut", "sturgeon", "barracuda", "swordfish", "marlin",
    "stingray", "piranha", "tadpole", "monitor", "komodo", "terrapin",
    "asp", "garter", "constrictor", "moray", "grouper", "snapper", "mullet",
    "haddock", "pollock", "tilapia",
]

CLUSTER_INSECTS = [
    # base-320: bee ant
    "bee", "ant", "wasp", "hornet", "horntail", "mosquito", "gnat", "midge",
    "beetle", "ladybug", "weevil", "cockroach", "termite", "cricket",
    "grasshopper", "locust", "mantis", "moth", "butterfly", "caterpillar",
    "dragonfly", "damselfly", "aphid", "flea", "louse", "tick", "mite",
    "spider", "scorpion", "centipede", "millipede", "worm", "slug", "snail",
    "earwig", "firefly", "cicada", "katydid", "bumblebee", "sawfly", "botfly",
    "larva", "grub", "maggot", "pupa", "silkworm", "bedbug", "stinkbug",
    "horsefly", "blowfly", "mayfly", "gadfly", "weaver", "tarantula",
    "harvestman", "springtail", "thrips", "lacewing", "antlion", "earworm",
    "chigger", "nit", "roach", "hopper",
]

CLUSTER_FRUITS = [
    # base-320: apple fruit
    "apple", "fruit", "banana", "orange", "pear", "peach", "plum", "cherry",
    "grape", "lemon", "lime", "melon", "berry", "strawberry", "raspberry",
    "blueberry", "blackberry", "cranberry", "mango", "papaya", "pineapple",
    "kiwi", "coconut", "apricot", "fig", "date", "olive", "grapefruit",
    "tangerine", "nectarine", "guava", "lychee", "pomegranate", "quince",
    "currant", "gooseberry", "mulberry", "elderberry", "avocado", "tomato",
    "honeyberry", "watermelon", "cantaloupe", "honeydew", "persimmon", "kumquat",
    "clementine", "mandarin", "passion", "dragonfruit", "starfruit",
    "jackfruit", "durian", "soursop", "boysenberry", "loganberry",
    "huckleberry", "plantain", "tamarind", "lingonberry", "cloudberry",
    "salmonberry", "damson", "sloe",
]

CLUSTER_VEGETABLES = [
    "carrot", "potato", "onion", "garlic", "pea", "bean", "corn", "lettuce",
    "cabbage", "spinach", "kale", "broccoli", "cauliflower", "celery",
    "cucumber", "pepper", "pumpkin", "squash", "zucchini", "eggplant",
    "radish", "turnip", "beet", "parsnip", "leek", "shallot", "chive",
    "asparagus", "artichoke", "mushroom", "okra", "yam", "cassava",
    "ginger", "horseradish", "rutabaga", "kohlrabi", "endive", "chard",
    "arugula", "watercress", "fennel", "chickpea", "lentil", "soybean",
    "edamame", "scallion", "sprout", "cress", "collard", "mustard",
    "rhubarb", "gourd", "marrow", "courgette", "aubergine", "swede",
    "mangetout", "sweetcorn", "beetroot", "salsify", "chicory", "samphire",
    "celeriac",
]

CLUSTER_PREPARED_FOODS = [
    "bread", "toast", "sandwich", "pizza", "pasta", "noodle", "rice",
    "soup", "stew", "salad", "burger", "taco", "burrito", "quiche", "pie",
    "cake", "cookie", "biscuit", "muffin", "pancake", "waffle", "donut",
    "pastry", "pudding", "custard", "jelly", "jam", "honey", "butter",
    "cheese", "yogurt", "cream", "egg", "omelet", "bacon", "sausage",
    "ham", "steak", "roast", "curry", "chili", "dumpling", "porridge",
    "cereal", "oatmeal", "gravy", "sauce", "dip", "spread", "dough",
    "crumb", "crust", "filling", "topping", "stuffing", "casserole",
    "fritter", "scone", "bagel", "pretzel", "cracker", "wafer", "candy",
    "chocolate",
]

CLUSTER_DRINKS = [
    # base-320: water
    "water", "milk", "juice", "tea", "coffee", "cocoa", "soda", "lemonade",
    "cider", "wine", "beer", "ale", "punch", "smoothie", "malt", "broth",
    "nectar", "syrup", "cola", "tonic", "espresso", "latte", "cappuccino",
    "mocha", "brew", "lager", "stout", "whiskey", "vodka", "rum", "gin",
    "brandy", "liquor", "cocktail", "champagne", "sherry", "vermouth", "mead",
    "kombucha", "kefir", "buttermilk", "eggnog", "slush", "sherbet",
    "barley", "cordial", "fizz", "sip", "gulp", "quencher", "beverage", "drip",
    "spritzer", "milkshake", "frappe", "infusion", "decaf", "brewage",
    "potion", "elixir", "draught", "swill", "grog", "sake",
]

CLUSTER_LAND_VEHICLES = [
    "car", "truck", "bus", "van", "taxi", "jeep", "wagon", "cart", "tractor",
    "trailer", "lorry", "bike", "bicycle", "motorcycle", "scooter", "moped",
    "train", "tram", "subway", "trolley", "carriage", "coach", "buggy",
    "sled", "sleigh", "skateboard", "rollerblade", "unicycle", "tricycle",
    "ambulance", "firetruck", "limousine", "minivan", "hatchback", "sedan",
    "coupe", "convertible", "pickup", "forklift", "bulldozer", "excavator",
    "hauler", "dumper", "roadster", "hearse", "chariot", "rickshaw", "minibus",
    "snowmobile", "segway", "monorail", "locomotive", "boxcar", "caboose",
    "handcart", "wheelbarrow", "pram", "stroller", "tandem", "snowplow",
    "racecar", "dragster", "rover", "trundle",
]

CLUSTER_AIR_WATER_VEHICLES = [
    "plane", "jet", "airplane", "helicopter", "glider", "blimp", "balloon",
    "airship", "rocket", "shuttle", "spacecraft", "satellite", "drone",
    "biplane", "seaplane", "airliner", "fighter", "bomber", "zeppelin",
    "parachute", "boat", "ship", "yacht", "canoe", "kayak", "raft", "barge",
    "ferry", "tanker", "tugboat", "trawler", "schooner", "sailboat",
    "steamer", "submarine", "freighter", "liner", "dinghy", "gondola",
    "houseboat", "catamaran", "galleon", "frigate", "destroyer", "cruiser",
    "battleship", "carrier", "sloop", "skiff", "punt", "rowboat", "speedboat",
    "hovercraft", "hydrofoil", "lifeboat", "junk", "dhow", "outrigger",
    "pontoon", "wherry", "cutter", "corvette", "ketch", "scow",
]

CLUSTER_HAND_TOOLS = [
    # base-320: key spoon
    "key", "spoon", "hammer", "screwdriver", "wrench", "pliers", "saw",
    "drill", "chisel", "file", "axe", "hatchet", "mallet", "knife", "blade",
    "scissors", "shears", "clippers", "tongs", "fork", "ladle", "spatula",
    "whisk", "peeler", "grater", "sieve", "strainer", "rake", "hoe",
    "shovel", "spade", "trowel", "pickaxe", "crowbar", "clamp", "vice",
    "ruler", "tape", "level", "compass", "pencil", "pen", "brush", "broom",
    "mop", "bucket", "scraper", "sander", "jigsaw", "awl", "bradawl", "gouge",
    "rasp", "needle", "pin", "nail", "screw", "bolt", "nut", "shim",
    "rivet", "stapler", "hook", "anchor",
]

CLUSTER_MACHINES = [
    "engine", "motor", "pump", "generator", "turbine", "compressor",
    "computer", "laptop", "tablet", "phone", "printer", "scanner",
    "camera", "radio", "television", "speaker", "console", "keyboard",
    "trackpad", "router", "modem", "battery", "charger", "fan", "heater",
    "furnace", "boiler", "refrigerator", "freezer", "oven", "stove",
    "microwave", "toaster", "blender", "mixer", "dishwasher", "washer",
    "dryer", "vacuum", "robot", "elevator", "escalator", "conveyor",
    "winch", "hoist", "lathe", "copier", "loom", "spindle", "gear",
    "piston", "valve", "switch", "circuit", "sensor", "thermostat",
    "projector", "amplifier", "transformer", "dynamo", "alternator",
    "windmill", "crusher", "machine",
]

CLUSTER_CLOTHING = [
    "shirt", "pants", "trousers", "jeans", "shorts", "skirt", "dress",
    "blouse", "sweater", "jacket", "coat", "vest", "hoodie", "cardigan",
    "blazer", "suit", "tie", "scarf", "hat", "cap", "beanie", "bonnet",
    "helmet", "glove", "mitten", "sock", "stocking", "shoe", "boot",
    "sandal", "slipper", "sneaker", "lace", "loafer", "belt", "buckle",
    "button", "zipper", "collar", "cuff", "sleeve", "hem", "pocket",
    "hood", "robe", "gown", "pajamas", "nightgown", "swimsuit", "bikini",
    "raincoat", "poncho", "cloak", "shawl", "apron", "uniform", "overalls",
    "leggings", "tights", "underwear", "bra", "garment", "fabric", "cloth",
]

CLUSTER_FURNITURE = [
    # base-320: chair table bed cup plate book ball box bag
    "chair", "table", "bed", "cup", "plate", "book", "ball", "box", "bag",
    "desk", "stool", "bench", "couch", "sofa", "armchair", "recliner",
    "ottoman", "shelf", "bookcase", "cabinet", "cupboard", "drawer",
    "dresser", "wardrobe", "closet", "nightstand", "bunk", "crib", "cradle",
    "mattress", "pillow", "blanket", "quilt", "cushion", "rug", "carpet",
    "mat", "lamp", "chandelier", "mirror", "frame", "clock", "vase",
    "basket", "bowl", "jar", "pot", "pan", "kettle", "tray", "mug", "glass",
    "bottle", "dish", "saucer", "napkin", "tablecloth", "curtain", "blind",
    "shutter", "easel", "rack", "hanger", "hamper",
]

CLUSTER_BUILDINGS = [
    # base-320: house school shop window door garden park road
    "house", "school", "shop", "window", "door", "garden", "park", "road",
    "home", "building", "store", "market", "mall", "office", "factory",
    "warehouse", "barn", "shed", "garage", "cabin", "cottage", "mansion",
    "castle", "palace", "tower", "church", "temple", "mosque", "cathedral",
    "chapel", "hospital", "clinic", "library", "museum", "theater",
    "stadium", "arena", "gym", "hotel", "motel", "inn", "restaurant",
    "cafe", "bakery", "bank", "prison", "jail", "station", "airport",
    "harbor", "port", "bridge", "tunnel", "fence", "gate", "wall", "roof",
    "floor", "ceiling", "stairs", "hallway", "room", "basement", "attic",
]

CLUSTER_BODY_PARTS = [
    # base-320: hand foot head eye arm leg ear nose mouth
    "hand", "foot", "head", "eye", "arm", "leg", "ear", "nose", "mouth",
    "face", "cheek", "chin", "jaw", "lip", "tongue", "tooth", "gum",
    "neck", "throat", "shoulder", "elbow", "wrist", "finger", "thumb",
    "cuticle", "palm", "knuckle", "knee", "ankle", "heel", "toe", "hip",
    "thigh", "shin", "calf", "rump", "spine", "chest", "rib", "belly",
    "stomach", "waist", "navel", "hair", "skin", "bone", "muscle", "heart",
    "lung", "liver", "kidney", "brain", "nerve", "vein", "blood", "skull",
    "forehead", "eyebrow", "eyelash", "nostril", "scalp", "limb", "torso",
    "tendon",
]

CLUSTER_PLANTS_TREES = [
    # base-320: tree flower leaf grass seed root branch
    "tree", "flower", "leaf", "grass", "seed", "root", "branch", "bush",
    "shrub", "vine", "weed", "fern", "moss", "ivy", "oak", "pine", "maple",
    "birch", "elm", "willow", "cedar", "spruce", "fir", "ash", "beech",
    "poplar", "sequoia", "bamboo", "cactus", "rose", "tulip", "daisy", "lily",
    "iris", "orchid", "sunflower", "poppy", "daffodil", "geranium",
    "jasmine", "marigold", "petunia", "clover", "thistle", "nettle",
    "reed", "bark", "twig", "stem", "petal", "bud", "blossom", "sapling",
    "trunk", "bough", "foliage", "thorn", "shoot", "seedling", "stump",
    "hedge", "thicket", "grove", "lichen",
]

CLUSTER_WEATHER_NATURE = [
    # base-320: fire sun moon
    "fire", "sun", "moon", "rain", "snow", "wind", "storm", "cloud", "fog",
    "mist", "frost", "ice", "hail", "thunder", "lightning", "rainbow",
    "sky", "star", "planet", "comet", "meteor", "sunshine", "sunset",
    "sunrise", "dawn", "dusk", "twilight", "shadow", "earthquake", "flood",
    "drought", "hurricane", "tornado", "blizzard", "breeze", "gust",
    "gale", "drizzle", "shower", "downpour", "sleet", "dew", "humidity",
    "heat", "chill", "warmth", "weather", "climate", "season", "spring",
    "summer", "autumn", "winter", "mountain", "valley", "hill", "cliff",
    "river", "desert", "forest", "ocean", "sea", "lake", "wave",
]

CLUSTER_KINSHIP_PEOPLE = [
    # base-320: person baby child friend mother father
    "person", "baby", "child", "friend", "mother", "father", "parent",
    "son", "daughter", "brother", "sister", "sibling", "grandfather",
    "grandmother", "grandparent", "grandson", "granddaughter", "uncle",
    "aunt", "nephew", "niece", "cousin", "husband", "wife", "spouse",
    "family", "relative", "kin", "ancestor", "descendant", "twin",
    "toddler", "infant", "kid", "boy", "girl", "man", "woman", "adult",
    "elder", "neighbor", "stranger", "guest", "host", "people", "crowd",
    "partner", "widow", "orphan", "guardian", "nanny", "teacher", "student",
    "doctor", "nurse", "farmer", "worker", "leader", "king", "queen",
    "prince", "princess", "citizen", "comrade",
]

CLUSTER_MOTION_VERBS = [
    # base-320: go come run walk jump fall fly swim sit stand turn climb
    # crawl ride throw catch kick hit stop
    "go", "come", "run", "walk", "jump", "fall", "fly", "swim", "sit",
    "stand", "turn", "climb", "crawl", "ride", "throw", "catch", "kick",
    "hit", "stop", "move", "dash", "sprint", "jog", "march", "stride",
    "skip", "hop", "leap", "bounce", "roll", "slide", "glide", "drift",
    "float", "sink", "rise", "dive", "plunge", "soar", "swoop", "dart",
    "race", "chase", "flee", "escape", "wander", "roam", "stroll", "pace",
    "tiptoe", "stagger", "stumble", "trip", "spin", "twirl", "swing",
    "sway", "rock", "shake", "tremble", "wobble", "lunge", "sneak",
    "scamper",
]

CLUSTER_PERCEPTION_VERBS = [
    # base-320: look see hear listen watch find touch smell taste feel
    "look", "see", "hear", "listen", "watch", "find", "touch", "smell",
    "taste", "feel", "observe", "notice", "spot", "glance", "stare",
    "gaze", "peek", "peer", "view", "witness", "perceive", "sense",
    "detect", "discover", "recognize", "identify", "examine", "inspect",
    "scan", "survey", "study", "scrutinize", "behold", "leer", "sight",
    "glimpse", "ogle", "squint", "blink", "wink", "sniff", "savor",
    "lick", "stroke", "rub", "pat", "poke", "prod", "grope", "fondle",
    "caress", "tickle", "graze", "skim", "handle", "grasp", "clutch",
    "supervise", "spy", "search", "seek", "discern", "regard", "overhear",
]

CLUSTER_COMMUNICATION_VERBS = [
    # base-320: speak read write say ask tell call answer
    "speak", "read", "write", "say", "ask", "tell", "call", "answer",
    "talk", "chat", "converse", "discuss", "explain", "describe", "state",
    "declare", "announce", "report", "mention", "remark", "comment",
    "reply", "respond", "shout", "yell", "whisper", "murmur", "mumble",
    "scream", "cheer", "sing", "chant", "hum", "recite", "narrate", "argue",
    "debate", "claim", "insist", "deny", "agree", "promise", "swear",
    "warn", "advise", "suggest", "command", "order", "request", "beg",
    "plead", "thank", "greet", "praise", "scold", "blame", "boast",
    "gossip", "rant", "lecture", "preach", "translate", "spell", "quote",
]

CLUSTER_MANIPULATION_VERBS = [
    # base-320: push pull open close give take hold drop cut break build
    # make fix carry bring send eat drink cook wash
    "push", "pull", "open", "close", "give", "take", "hold", "drop", "cut",
    "break", "build", "make", "fix", "carry", "bring", "send", "eat",
    "drink", "cook", "wash", "lift", "lower", "press", "squeeze", "twist",
    "bend", "fold", "tear", "rip", "snap", "crush", "smash", "crack",
    "split", "slice", "chop", "grind", "stir", "mix", "pour", "fill",
    "drain", "wipe", "scrub", "rinse", "dab", "sweep", "polish", "paint",
    "glue", "stick", "bind", "untie", "knot", "wrap", "pack", "stack",
    "pile", "sort", "arrange", "place", "remove", "grab", "shove",
]

CLUSTER_EMOTION_STATES = [
    # base-320: happy sad angry scared laugh cry play work wait wake sleep
    "happy", "sad", "angry", "scared", "laugh", "cry", "play", "work",
    "wait", "wake", "sleep", "joy", "grief", "fear", "anger", "love",
    "hate", "hope", "worry", "calm", "excited", "bored", "tired", "weary",
    "afraid", "nervous", "anxious", "glad", "cheerful", "gloomy", "moody",
    "lonely", "jealous", "proud", "ashamed", "guilty", "shy", "brave",
    "eager", "content", "restless", "frustrated", "annoyed", "furious",
    "delighted", "miserable", "depressed", "hopeful", "grateful", "smile",
    "frown", "weep", "sob", "sigh", "groan", "rejoice", "mourn", "panic",
    "relax", "rest", "yawn", "doze", "dream", "fret",
]

CLUSTER_SIZE_SHAPE_ADJ = [
    # base-320: big small tall short long wide huge tiny thin thick deep
    # narrow round flat sharp
    "big", "small", "tall", "short", "long", "wide", "huge", "tiny", "thin",
    "thick", "deep", "narrow", "round", "flat", "sharp", "large", "little",
    "giant", "massive", "enormous", "vast", "teeny", "miniature", "broad",
    "slim", "slender", "chunky", "plump", "shallow", "high", "low", "steep",
    "square", "circular", "oval", "curved", "straight", "crooked", "bent",
    "pointed", "blunt", "jagged", "smooth", "bumpy", "lumpy", "hollow",
    "solid", "compact", "bulky", "gigantic", "colossal", "petite", "lanky",
    "squat", "wiry", "spherical", "cylindrical", "rectangular", "triangular",
    "angular", "tapered", "domed", "convex", "concave",
]

CLUSTER_COLOR_ADJ = [
    # base-320: red blue green yellow white black bright dark
    "red", "blue", "green", "yellow", "white", "black", "bright", "dark",
    "ochre", "purple", "pink", "brown", "gray", "violet", "indigo",
    "crimson", "scarlet", "maroon", "ruby", "blush", "coral", "rosy",
    "amber", "gold", "golden", "tan", "beige", "eggshell", "ivory", "khaki",
    "hunter", "chartreuse", "emerald", "jade", "teal", "turquoise", "aqua", "cyan",
    "navy", "azure", "cobalt", "lavender", "lilac", "magenta", "fuchsia",
    "burgundy", "mauve", "bronze", "copper", "rust", "chestnut", "auburn",
    "silver", "charcoal", "ebony", "pale", "pastel", "vivid", "colorful",
    "dim", "shiny", "glossy", "matte", "translucent",
]

CLUSTER_TEXTURE_MATERIAL_ADJ = [
    # base-320: hot cold warm cool fast slow new old clean dirty wet dry
    # hard soft full empty heavy light sweet sour strong weak rich poor
    # kind mean nice good bad young sick well true false
    "hot", "cold", "warm", "cool", "fast", "slow", "new", "old", "clean",
    "dirty", "wet", "dry", "hard", "soft", "full", "empty", "heavy",
    "light", "sweet", "sour", "strong", "weak", "rich", "poor", "kind",
    "mean", "nice", "good", "bad", "young", "sick", "well", "true",
    "false", "rough", "fuzzy", "silky", "fluffy", "coarse", "grainy",
    "sticky", "slippery", "greasy", "oily", "crisp", "brittle", "tough",
    "tender", "firm", "rigid", "flexible", "stiff", "elastic", "spongy",
    "metallic", "wooden", "plastic", "glassy", "rubbery", "leathery",
    "woolly", "loud", "damp", "quiet",
]

CLUSTER_TIME_WORDS = [
    # base-320: now then before after first last next today yesterday
    # tomorrow soon late early always never often sometimes once again
    # later until since during whenever
    "now", "then", "before", "after", "first", "last", "next", "today",
    "yesterday", "tomorrow", "soon", "late", "early", "always", "never",
    "often", "sometimes", "once", "again", "later", "until", "since",
    "during", "whenever", "morning", "noon", "afternoon", "evening",
    "night", "midnight", "day", "week", "month", "year", "hour", "minute",
    "second", "moment", "instant", "while", "meanwhile", "afterward",
    "beforehand", "previously", "currently", "presently", "eventually",
    "finally", "immediately", "instantly", "shortly", "recently", "lately",
    "frequently", "rarely", "seldom", "occasionally", "daily", "weekly",
    "monthly", "yearly", "hourly", "henceforth", "nowadays",
]

CLUSTER_SPATIAL_WORDS = [
    # base-320: north south east west up down left right here there near
    # far in out on under above below front back top bottom side middle
    # inside outside between around through across along toward away beside
    # behind beyond center edge corner forward
    "north", "south", "east", "west", "up", "down", "left", "right",
    "here", "there", "near", "far", "in", "out", "on", "under", "above",
    "below", "front", "back", "top", "bottom", "side", "middle", "inside",
    "outside", "between", "around", "through", "across", "along", "toward",
    "away", "beside", "behind", "beyond", "center", "edge", "corner",
    "forward", "backward", "upward", "downward", "inward", "outward",
    "sideways", "underneath", "overhead", "beneath", "atop", "within",
    "amid", "among", "alongside", "opposite", "adjacent", "nearby",
    "distant", "everywhere", "nowhere", "somewhere", "anywhere", "off",
    "onto",
]

CLUSTER_QUANTITY_NUMBER_WORDS = [
    # base-320: one two three four five six seven eight nine ten zero half
    # both many few some all none every any each
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "zero", "half", "both", "many", "few", "some", "all", "none",
    "every", "any", "each", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "dozen", "pair", "couple",
    "several", "plenty", "much", "more", "most", "less", "least", "enough",
    "single", "double", "triple", "quarter", "third", "whole", "extra",
    "amount", "number", "total", "count",
]

CLUSTER_QUESTION_DISCOURSE = [
    # base-320: what where when who why how which whose yes no please thanks
    # hello goodbye sorry ok this that these those it
    "what", "where", "when", "who", "why", "how", "which", "whose", "yes",
    "no", "please", "thanks", "hello", "goodbye", "sorry", "ok", "this",
    "that", "these", "those", "it", "whom", "whatever", "wherever",
    "whoever", "however", "whichever", "okay", "yeah", "yep", "nope",
    "hey", "hi", "bye", "farewell", "welcome", "greetings", "cheers",
    "congratulations", "oops", "ouch", "wow", "oh", "ah", "huh", "hmm",
    "uh", "righto", "indeed", "certainly", "absolutely", "exactly",
    "maybe", "perhaps", "pardon", "excuse", "regret", "apology", "alright",
    "anyway", "anyhow", "namely", "regardless", "uhh",
]

CLUSTER_ABSTRACT_RELATIONS = [
    # base-320: and or but if because can will do did is have want need
    # very too also only not same other another
    "and", "or", "but", "if", "because", "can", "will", "do", "did", "is",
    "have", "want", "need", "very", "too", "also", "only", "not", "same",
    "other", "another", "cause", "reason", "result", "therefore", "thus",
    "hence", "consequently", "although", "though", "unless", "whether",
    "given", "whereas", "owing", "so", "yet", "nor", "either", "neither",
    "regarding", "instead", "besides", "moreover", "furthermore", "nonetheless",
    "meaning", "purpose", "effect", "factor", "condition", "difference",
    "similarity", "relation", "connection", "contrast", "comparison",
    "consequence", "implication", "must", "should", "would", "could",
    "might",
]

# ---------------------------------------------------------------------------
# Assemble the 32-cluster mapping (ordered exactly per the spec).
# ---------------------------------------------------------------------------

ALL_CLUSTERS_2048: dict[str, list[str]] = {
    "mammals": CLUSTER_MAMMALS,
    "birds": CLUSTER_BIRDS,
    "fish_reptiles": CLUSTER_FISH_REPTILES,
    "insects": CLUSTER_INSECTS,
    "fruits": CLUSTER_FRUITS,
    "vegetables": CLUSTER_VEGETABLES,
    "prepared_foods": CLUSTER_PREPARED_FOODS,
    "drinks": CLUSTER_DRINKS,
    "land_vehicles": CLUSTER_LAND_VEHICLES,
    "air_water_vehicles": CLUSTER_AIR_WATER_VEHICLES,
    "hand_tools": CLUSTER_HAND_TOOLS,
    "machines": CLUSTER_MACHINES,
    "clothing": CLUSTER_CLOTHING,
    "furniture": CLUSTER_FURNITURE,
    "buildings": CLUSTER_BUILDINGS,
    "body_parts": CLUSTER_BODY_PARTS,
    "plants_trees": CLUSTER_PLANTS_TREES,
    "weather_nature": CLUSTER_WEATHER_NATURE,
    "kinship_people": CLUSTER_KINSHIP_PEOPLE,
    "motion_verbs": CLUSTER_MOTION_VERBS,
    "perception_verbs": CLUSTER_PERCEPTION_VERBS,
    "communication_verbs": CLUSTER_COMMUNICATION_VERBS,
    "manipulation_verbs": CLUSTER_MANIPULATION_VERBS,
    "emotion_states": CLUSTER_EMOTION_STATES,
    "size_shape_adj": CLUSTER_SIZE_SHAPE_ADJ,
    "color_adj": CLUSTER_COLOR_ADJ,
    "texture_material_adj": CLUSTER_TEXTURE_MATERIAL_ADJ,
    "time_words": CLUSTER_TIME_WORDS,
    "spatial_words": CLUSTER_SPATIAL_WORDS,
    "quantity_number_words": CLUSTER_QUANTITY_NUMBER_WORDS,
    "question_discourse": CLUSTER_QUESTION_DISCOURSE,
    "abstract_relations": CLUSTER_ABSTRACT_RELATIONS,
}

# Per-cluster invariants (run AT IMPORT).
for _name, _v in ALL_CLUSTERS_2048.items():
    assert len(_v) == 64, f"{_name} has {len(_v)} concepts, expected 64"
    assert len(_v) == len(set(_v)), (
        f"{_name} has internal duplicates: "
        f"{sorted(w for w in _v if _v.count(w) > 1)}"
    )

# Flat list of all 2048 concept words.
ALL_WORDS_2048: list[str] = []
for _v in ALL_CLUSTERS_2048.values():
    ALL_WORDS_2048.extend(_v)

# Safety net: ANY cross-cluster collision in the hand-curation fails HERE
# at import (and in the test) -- never silently.
assert len(ALL_WORDS_2048) == len(set(ALL_WORDS_2048)), (
    "Duplicate words across clusters: "
    f"{sorted(w for w in ALL_WORDS_2048 if ALL_WORDS_2048.count(w) > 1)}"
)

TOTAL_VOCAB_2048 = 2048
assert len(ALL_WORDS_2048) == TOTAL_VOCAB_2048, \
    f"Total vocab size {len(ALL_WORDS_2048)}, expected {TOTAL_VOCAB_2048}"


def write_vocab_files_2048(out_dir: str = "research/findings/raw/g11_bg"):
    """Write a 64-concept vocab file per cluster under a distinct
    g20_<cluster>_vocab2048.txt name so the validated 160/320-concept
    vocab files (and the trained bridges) are NOT clobbered."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, vocab in ALL_CLUSTERS_2048.items():
        path = out / f"g20_{name}_vocab2048.txt"
        path.write_text("\n".join(vocab))
        print(f"  wrote {path}: {len(vocab)} concepts")
        written.append(str(path))
    return written


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true",
                    help="Write 64-concept vocab files (vocab2048.txt)")
    p.add_argument("--out-dir", type=str,
                    default="research/findings/raw/g11_bg")
    args = p.parse_args()
    print(f"32-cluster G.20 2048-concept vocab spec "
          f"({TOTAL_VOCAB_2048} unique concepts, 64/cluster):")
    for name, vocab in ALL_CLUSTERS_2048.items():
        print(f"  {name:>22} ({len(vocab)}): "
              f"{vocab[:4]} ... {vocab[-3:]}")
    print(f"\nTotal: {len(ALL_WORDS_2048)} unique words, "
          f"no duplicates across clusters.")
    if args.write:
        print(f"\nWriting 64-concept vocab files to {args.out_dir}:")
        write_vocab_files_2048(args.out_dir)
