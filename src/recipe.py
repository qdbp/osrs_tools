from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Item:
    id: int
    name: str
    limit: Optional[int]
    value: int


@dataclass()
class Recipe:
    
    ingredients: dict[str, int]
    products: dict[str, int]
    name: str = None

    def __post_init__(self):
        if self.name is None:
            self.name = list(self.products.keys())[0]


BWOOL = "Ball of wool"

GEM_RT = "Red topaz"
GEM_OP = "Opal"
GEM_JD = "Jade"

BAR_AG = "Silver bar"
BAR_AU = "Gold bar"

RU_COS = "Cosmic rune"
RU_NAT = "Nature rune"

JWL_DODGE = "Dodgy necklace"
JWL_BURN = "Burning amulet(5)"
JWL_CHEM = "Amulet of chemistry"

JEWEL_RECIPES = [
    Recipe(
        ingredients={BAR_AG: 1, RU_COS: 1, GEM_OP: 1}, products={JWL_DODGE: 1}
    ),
    Recipe(
        ingredients={BAR_AG: 1, RU_COS: 1, GEM_RT: 1, BWOOL: 1},
        products={JWL_BURN: 1},
    ),
    Recipe(
        ingredients={BAR_AG: 1, RU_COS: 1, GEM_JD: 1, BWOOL: 1},
        products={JWL_CHEM: 1},
    ),
]

HERBS = [
    "Guam leaf",
    "Marrentill",
    "Tarromin",
    "Harralander",
    "Ranarr weed",
    "Toadflax",
    "Irit leaf",
    "Avantoe",
    "Kwuarm",
    "Snapdragon",
    "Cadantine",
    "Lantadyme",
    "Dwarf weed",
    "Torstol"
]

HERB_RECIPES = [
    Recipe(
        ingredients={f'Grimy {herb.lower()}': 1},
        products={herb: 1}
    )
    for herb in HERBS
]

# omits vial for simplicity
SIMPLE_POT_RECIPES = [
    Recipe(
        ingredients={'Ranarr weed': 1, 'Snape grass': 1},
        products={'Prayer potion(3)': 1}
    )
]
