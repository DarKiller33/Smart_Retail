# product_db.py
# Retail SKU Catalog & Allowed Classes Whitelist

# EXPLICIT WHITELIST: Only these items are recognized. 'person' and background objects are ignored.
ALLOWED_RETAIL_CLASSES = {"apple", "banana", "orange", "bottle", "cup", "cell phone"}

PRODUCT_CATALOG = {
    "apple": {"name": "Fresh Red Apple", "category": "Produce", "price": 30.0, "unit_weight_g": 180.0, "epc": "EPC_APPLE_RED"},
    "banana": {"name": "Robusta Banana", "category": "Produce", "price": 10.0, "unit_weight_g": 120.0, "epc": "EPC_BANANA"},
    "orange": {"name": "Nagpur Orange", "category": "Produce", "price": 25.0, "unit_weight_g": 150.0, "epc": "EPC_ORANGE"},
    "bottle": {"name": "Bisleri Water 1L", "category": "Beverages", "price": 20.0, "unit_weight_g": 1000.0, "epc": "EPC_BISLERI_1L"},
    "cup": {"name": "Nescafe Coffee Cup", "category": "Beverages", "price": 15.0, "unit_weight_g": 250.0, "epc": "EPC_COFFEE_CUP"},
    "cell phone": {"name": "Mobile Phone", "category": "Electronics", "price": 15000.0, "unit_weight_g": 200.0, "epc": "EPC_MOBILE_PHONE"}
}

def get_product_metadata(label: str) -> dict:
    clean_key = str(label).strip().lower()
    
    if clean_key in PRODUCT_CATALOG:
        return PRODUCT_CATALOG[clean_key]
    
    formatted_name = clean_key.replace("_", " ").title()
    return {
        "name": formatted_name,
        "category": "Retail SKU",
        "price": 40.0,
        "unit_weight_g": 250.0,
        "epc": f"EPC_{clean_key.upper()}"
    }