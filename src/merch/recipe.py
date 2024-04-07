from dataclasses import dataclass
from math import floor
from typing import Optional


BANK_OVERHEAD = 3


@dataclass(frozen=True)
class Item:
    id: int
    name: str
    limit: Optional[int]
    value: int


@dataclass()
class Recipe:
    name: str
    ingredients: dict[str, int]
    products: dict[str, float]


@dataclass()
class BankableRecipe(Recipe):
    """
    Represents a recipe that can be crafted at a bank, with the only overhead being
    depositing and withdrawing items.
    """
    inv_size: int
    ticks: int

    def pps(self, n_ops: int):
        """
        Products per second.
        """
        assert n_ops >= 1
        n_loads = (n_ops - 1) // self.inv_size + 1
        return n_loads * (BANK_OVERHEAD + self.inv_size * 0.6 * self.ticks)


BWOOL = "Ball of wool"

GEM_RT = "Red topaz"
GEM_OP = "Opal"
GEM_JD = "Jade"

BAR_AG = "Silver bar"
BAR_AU = "Gold bar"

RU_COS = "Cosmic rune"
RU_NAT = "Nature rune"

JWL_DODGE = "Dodgy necklace"
JWL_CHEM = "Amulet of chemistry"
JWL_PASS = "Necklace of passage(5)"
JWL_BURN = "Burning amulet(5)"

JEWEL_RECIPES = [
    Recipe(
        JWL_DODGE,
        ingredients={BAR_AG: 1, RU_COS: 1, GEM_OP: 1},
        products={JWL_DODGE: 1},
    ),
    Recipe(
        JWL_CHEM,
        ingredients={BAR_AG: 1, RU_COS: 1, GEM_JD: 1, BWOOL: 1},
        products={JWL_CHEM: 1},
    ),
    Recipe(
        JWL_PASS,
        ingredients={BAR_AG: 1, RU_COS: 1, GEM_JD: 1},
        products={JWL_PASS: 1},
    ),
    Recipe(
        JWL_BURN,
        ingredients={BAR_AG: 1, RU_COS: 1, GEM_RT: 1, BWOOL: 1},
        products={JWL_BURN: 1},
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
    "Torstol",
]

HERB_RECIPES = [
    Recipe(
        f"Clean {herb}", ingredients={f"Grimy {herb.lower()}": 1}, products={herb: 1}
    )
    for herb in HERBS
]

# omits vial for simplicity
SIMPLE_POT_RECIPES = [
    Recipe(
        "Make prayer pot",
        ingredients={"Ranarr weed": 1, "Snape grass": 1},
        products={"Prayer potion(3)": 1},
    )
]


def get_opal_recipe(crft: int) -> Recipe:
    return BankableRecipe(
        "Cut opal",
        ingredients={"Uncut opal": 1},
        products={"Opal": floor((crft - 1) * (122 / 98) + 129) / 256},
        ticks=2,
        inv_size=27,
    )


def get_jade_recipe(crft: int) -> Recipe:
    return BankableRecipe(
        "Cut jade",
        ingredients={"Uncut jade": 1},
        products={"Jade": floor((crft - 1) * (145 / 98) + 101) / 256},
        ticks=2,
        inv_size=27,
    )


def get_rtopaz_recipe(crft: int) -> Recipe:
    return BankableRecipe(
        "Cut red topaz",
        ingredients={"Uncut red topaz": 1},
        products={"Red topaz": floor((crft - 1) * (150 / 98) + 91) / 256},
        ticks=2,
        inv_size=27,
    )
