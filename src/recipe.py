from dataclasses import dataclass


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
JWL_BURN = "Burning amulet"
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
