import time

class ShoppingCartManager:
    def __init__(self, debounce_frames: int = 5):
        """
        Manages persistent shopping cart state, item debouncing, and billing totals.
        """
        self.cart_ledger = {}  # {sku_or_label: {"name": str, "qty": int, "price": float, "source": str, "status": str}}
        self.detection_history = {} # Track consecutive detections to avoid noise
        self.debounce_frames = debounce_frames

    def add_item_by_barcode(self, product_info: dict):
        """
        Instantly adds an item to the cart when a valid barcode is scanned.
        """
        name = product_info["name"]
        price = product_info["price"]
        
        if name in self.cart_ledger:
            self.cart_ledger[name]["qty"] += 1
        else:
            self.cart_ledger[name] = {
                "name": name,
                "qty": 1,
                "price": price,
                "source": "BARCODE_SCAN",
                "status": "VERIFIED_MATCH"
            }
        print(f"[CART EVENT] Barcode Scanned: Added '{name}' (₹{price})")

    def update_fused_state(self, fused_inventory: dict):
        """
        Updates cart items based on persistent multi-modal fusion reads.
        """
        for item_label, info in fused_inventory.items():
            qty = info.get("verified_qty", max(info["visual_qty"], info["rfid_qty"]))
            if qty > 0:
                # Update consecutive detection counter
                self.detection_history[item_label] = self.detection_history.get(item_label, 0) + 1
                
                # Add to ledger once detection passes temporal debounce threshold
                if self.detection_history[item_label] >= self.debounce_frames:
                    if item_label not in self.cart_ledger or self.cart_ledger[item_label]["source"] != "BARCODE_SCAN":
                        self.cart_ledger[item_label] = {
                            "name": info.get("name", item_label.capitalize()),
                            "qty": qty,
                            "price": info.get("price", 50.0),
                            "source": "MULTI_MODAL_FUSION",
                            "status": info["status"]
                        }

    def clear_cart(self):
        """Resets trolley cart state to zero items."""
        self.cart_ledger.clear()
        self.detection_history.clear()

    def calculate_bill(self):
        """Calculates Subtotal, 18% GST, and Grand Total in INR."""
        subtotal = sum(item["price"] * item["qty"] for item in self.cart_ledger.values())
        gst = subtotal * 0.18
        grand_total = subtotal + gst
        return subtotal, gst, grand_total