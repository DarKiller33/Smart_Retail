# eco_health_engine.py
# Real-time Eco-Footprint & Health Profile Intelligence

PRODUCT_ECO_HEALTH_DB = {
    "Amul Taaza Milk 500ml": {"co2_g": 320, "sugar_g": 12.0, "allergens": ["dairy"]},
    "Lays Salted Chips 50g": {"co2_g": 180, "sugar_g": 1.2, "allergens": ["gluten"]},
    "Bisleri Mineral Water 1L": {"co2_g": 90, "sugar_g": 0.0, "allergens": []},
    "Cadbury Dairy Milk 100g": {"co2_g": 450, "sugar_g": 56.0, "allergens": ["dairy", "nuts"]}
}

class EcoHealthAnalytics:
    def __init__(self, user_allergies=["nuts"]):
        self.user_allergies = user_allergies

    def analyze_basket(self, cart_items: dict) -> dict:
        total_co2_g = 0.0
        total_sugar_g = 0.0
        allergen_warnings = []

        for item_name, info in cart_items.items():
            qty = info.get("qty", 1)
            meta = PRODUCT_ECO_HEALTH_DB.get(item_name, {"co2_g": 150, "sugar_g": 5.0, "allergens": []})

            total_co2_g += (meta["co2_g"] * qty)
            total_sugar_g += (meta["sugar_g"] * qty)

            for allergen in meta["allergens"]:
                if allergen in self.user_allergies:
                    allergen_warnings.append(f"WARNING: '{item_name}' contains {allergen.upper()}!")

        return {
            "total_co2_kg": total_co2_g / 1000.0,
            "total_sugar_g": total_sugar_g,
            "warnings": list(set(allergen_warnings))
        }