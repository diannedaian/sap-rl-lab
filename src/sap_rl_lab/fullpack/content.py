"""Version-pinned late-game content. Sources/uncertainties: FULLPACK_RULES.md.

This catalog is an executable specification, not evidence of client parity.
"""

from dataclasses import asdict

from ..catalog import load_catalog as load_historical


def pet(name, atk, hp, trigger, effect, **params):
    return dict(
        id=name,
        name=name.title(),
        tier=6,
        attack=atk,
        health=hp,
        abilities=[dict(trigger=trigger, effect=effect, **params)],
    )


LATE_PETS = (
    pet(
        "boar",
        10,
        6,
        "before_attack",
        "buff_self",
        attack_by_level=[4, 8, 12],
        health_by_level=[2, 4, 6],
    ),
    pet(
        "cat",
        4,
        5,
        "shop_food",
        "multiply_food",
        multiplier_by_level=[2, 3, 4],
        max_uses=2,
        merge_trigger_rule="target-reset-on-level-up",
    ),
    pet(
        "dragon",
        3,
        8,
        "friend_bought",
        "buff_random_friend",
        count=5,
        attack_by_level=[1, 2, 3],
        health_by_level=[1, 2, 3],
        subject_tier=1,
        max_uses=4,
        merge_trigger_rule="target-reset-on-level-up",
    ),
    pet(
        "fly",
        4,
        4,
        "friend_faints",
        "fly_summon",
        summon_id="zombie_fly",
        summon_attack_by_level=[4, 8, 12],
        summon_health_by_level=[4, 8, 12],
        max_uses=3,
        merge_trigger_rule="target-reset-on-level-up",
    ),
    pet(
        "gorilla",
        7,
        10,
        "hurt",
        "gain_coconut",
        max_uses_by_level=[1, 2, 3],
        merge_trigger_rule="target-reset-on-level-up",
    ),
    pet(
        "leopard",
        10,
        4,
        "start_battle",
        "damage_attack_percent",
        percent=50,
        count_by_level=[1, 2, 3],
        rounding="ceil",
    ),
    pet(
        "mammoth",
        4,
        12,
        "faint",
        "buff_random_friend",
        count=5,
        attack_by_level=[2, 4, 6],
        health_by_level=[2, 4, 6],
    ),
    pet("piranha", 10, 4, "hurt", "buff_random_friend", count=5, attack_by_level=[3, 6, 9]),
    pet(
        "snake",
        8,
        3,
        "friend_ahead_attacks",
        "damage_random_enemy",
        count=1,
        damage_by_level=[5, 10, 15],
        max_uses=5,
        quota_scope="battle",
        merge_trigger_rule="target-reset-on-level-up",
    ),
    pet("tiger", 6, 4, "friend_ability", "repeat_ahead"),
    dict(id="zombie_fly", name="Zombie Fly", tier=0, attack=4, health=4, token=True, abilities=[]),
)

LATE_FOODS = (
    dict(id="chili", name="Chili", tier=5, cost=3, effect="set_perk", perk="chili"),
    dict(id="chocolate", name="Chocolate", tier=5, cost=3, effect="gain_experience", experience=1),
    dict(
        id="sushi",
        name="Sushi",
        tier=5,
        cost=3,
        effect="buff_random_team",
        attack=1,
        health=1,
        count=3,
    ),
    dict(id="melon", name="Melon", tier=6, cost=3, effect="set_perk", perk="melon"),
    dict(id="mushroom", name="Mushroom", tier=6, cost=3, effect="set_perk", perk="mushroom"),
    dict(
        id="pizza",
        name="Pizza",
        tier=6,
        cost=3,
        effect="buff_random_team",
        attack=2,
        health=2,
        count=2,
    ),
    dict(id="steak", name="Steak", tier=6, cost=3, effect="set_perk", perk="steak"),
    *(
        dict(
            id=name,
            name=name.replace("_", " ").title(),
            tier=1,
            cost=0,
            effect="buff",
            attack=level,
            health=0,
            token=True,
        )
        for level, name in enumerate(
            ("chocolate_milk", "better_chocolate_milk", "best_chocolate_milk"), 1
        )
    ),
)


def catalog_data(shop_tier=6):
    if shop_tier not in (5, 6):
        raise ValueError("Full-pack curriculum must have Tier5 or Tier6 normal shops")
    base = asdict(load_historical("turtle_v0_46_tier4_v6.json"))
    pets = []
    for item in base["pets"].values():
        pets.append(
            {
                **item,
                "abilities": [
                    {"trigger": a["trigger"], "effect": a["effect"], **a["params"]}
                    for a in item["abilities"]
                ],
            }
        )
    foods = [
        {**{k: v for k, v in f.items() if k != "params"}, **f["params"]}
        for f in base["foods"].values()
    ]
    return {
        **base,
        "catalog_id": f"turtle-v0.46-{'tier5-v7' if shop_tier == 5 else 'full-v8'}",
        "rules_version": 7 if shop_tier == 5 else 8,
        "checked_on": "2026-09-17",
        "development_only": True,
        "implemented_shop_tiers": [1, 2, 3, 4, 5, 6],
        "pets": pets + list(LATE_PETS),
        "foods": foods + list(LATE_FOODS),
    }
