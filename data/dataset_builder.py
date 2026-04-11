"""
Dataset Builder — Multi-Level Product Taxonomy
===============================================
Creates a labeled dataset of product titles mapped to 3-level category paths.

The taxonomy tree looks like:
    Level 1 (broad)    → Level 2 (mid)        → Level 3 (specific)
    Electronics        → Audio                 → Headphones
    Electronics        → Phones & Accessories  → Smartphones
    Clothing           → Men's Clothing        → Jeans
    ...

Each product gets a FULL PATH like: "Electronics > Audio > Headphones"
The model learns to predict all 3 levels at once from the title alone.
"""

import pandas as pd
import random

# ─────────────────────────────────────────────────────────────
# TAXONOMY TREE: 5 Level-1 categories, each with sub-categories
# ─────────────────────────────────────────────────────────────
PRODUCTS = [

    # ── ELECTRONICS > Audio > Headphones ──────────────────────
    ("Sony WH-1000XM5 Wireless Noise Cancelling Headphones Black", "Electronics", "Audio", "Headphones"),
    ("Bose QuietComfort 45 Bluetooth Headphones White", "Electronics", "Audio", "Headphones"),
    ("Apple AirPods Pro 2nd Generation MagSafe Case", "Electronics", "Audio", "Headphones"),
    ("Jabra Evolve2 85 Wireless Headset Black", "Electronics", "Audio", "Headphones"),
    ("Sennheiser HD 600 Open-Back Wired Headphones", "Electronics", "Audio", "Headphones"),
    ("JBL Tune 760NC Wireless Over-Ear Headphones Blue", "Electronics", "Audio", "Headphones"),

    # ── ELECTRONICS > Audio > Speakers ────────────────────────
    ("Bose SoundLink Flex Bluetooth Portable Speaker", "Electronics", "Audio", "Speakers"),
    ("JBL Charge 5 Portable Waterproof Bluetooth Speaker Black", "Electronics", "Audio", "Speakers"),
    ("Sonos Era 300 Smart Speaker White", "Electronics", "Audio", "Speakers"),
    ("Amazon Echo Dot 5th Generation Smart Speaker Charcoal", "Electronics", "Audio", "Speakers"),
    ("Marshall Stanmore III Bluetooth Home Speaker Black", "Electronics", "Audio", "Speakers"),

    # ── ELECTRONICS > Phones > Smartphones ────────────────────
    ("Apple iPhone 15 Pro Max 256GB Natural Titanium Unlocked", "Electronics", "Phones", "Smartphones"),
    ("Samsung Galaxy S24 Ultra 512GB Titanium Black", "Electronics", "Phones", "Smartphones"),
    ("Google Pixel 8 Pro 128GB Obsidian Unlocked", "Electronics", "Phones", "Smartphones"),
    ("OnePlus 12 256GB Silky Black 5G Unlocked", "Electronics", "Phones", "Smartphones"),
    ("Apple iPhone 14 128GB Midnight Black Unlocked", "Electronics", "Phones", "Smartphones"),

    # ── ELECTRONICS > Phones > Phone Cases ────────────────────
    ("OtterBox Defender Series Case for iPhone 15 Pro Black", "Electronics", "Phones", "Phone Cases"),
    ("Spigen Tough Armor Case for Samsung Galaxy S24 Ultra", "Electronics", "Phones", "Phone Cases"),
    ("Apple MagSafe Clear Case for iPhone 15", "Electronics", "Phones", "Phone Cases"),
    ("Casetify Ultra Impact Case iPhone 15 Pro Max", "Electronics", "Phones", "Phone Cases"),

    # ── ELECTRONICS > Computers > Laptops ─────────────────────
    ("Apple MacBook Pro 14-inch M3 Pro 512GB Space Black", "Electronics", "Computers", "Laptops"),
    ("Dell XPS 15 Intel Core i9 32GB RAM 1TB SSD", "Electronics", "Computers", "Laptops"),
    ("Lenovo ThinkPad X1 Carbon Gen 12 14in Intel i7 16GB", "Electronics", "Computers", "Laptops"),
    ("ASUS ROG Zephyrus G14 AMD Ryzen 9 RTX 4060 16GB", "Electronics", "Computers", "Laptops"),
    ("Microsoft Surface Pro 9 Intel i5 16GB 256GB Platinum", "Electronics", "Computers", "Laptops"),

    # ── ELECTRONICS > Computers > Monitors ────────────────────
    ("LG 27GP850-B 27-inch QHD 180Hz Gaming Monitor", "Electronics", "Computers", "Monitors"),
    ("Dell UltraSharp U2723D 27in 4K USB-C Monitor", "Electronics", "Computers", "Monitors"),
    ("Samsung Odyssey G7 32in QHD 240Hz Curved Gaming Monitor", "Electronics", "Computers", "Monitors"),
    ("BenQ PD2705U 27in 4K IPS Designer Monitor", "Electronics", "Computers", "Monitors"),

    # ── CLOTHING > Men's Clothing > Jeans ─────────────────────
    ("Levi's 501 Original Fit Jeans Men's 32x32 Dark Stonewash", "Clothing", "Men's Clothing", "Jeans"),
    ("Wrangler Cowboy Cut Slim Fit Jeans Men's 34x30 Blue", "Clothing", "Men's Clothing", "Jeans"),
    ("Diesel D-Luster Slim Fit Jeans Men's 33x32 Indigo", "Clothing", "Men's Clothing", "Jeans"),
    ("Calvin Klein Slim Taper Jeans Men's 30x30 Dark Wash", "Clothing", "Men's Clothing", "Jeans"),

    # ── CLOTHING > Men's Clothing > T-Shirts ──────────────────
    ("Hanes Men's ComfortSoft Crewneck T-Shirt White 3-Pack L", "Clothing", "Men's Clothing", "T-Shirts"),
    ("Nike Dri-FIT Men's Training T-Shirt Black Large", "Clothing", "Men's Clothing", "T-Shirts"),
    ("Ralph Lauren Polo Men's Custom Slim Fit T-Shirt Navy M", "Clothing", "Men's Clothing", "T-Shirts"),
    ("Carhartt Men's Workwear Pocket T-Shirt Dark Grey XL", "Clothing", "Men's Clothing", "T-Shirts"),

    # ── CLOTHING > Women's Clothing > Dresses ─────────────────
    ("Anthropologie Embroidered Maxi Dress Women's S Blue", "Clothing", "Women's Clothing", "Dresses"),
    ("ASOS Design Satin Midi Slip Dress Women's 10 Blush", "Clothing", "Women's Clothing", "Dresses"),
    ("Zara Floral Print Wrap Midi Dress Women's M Green", "Clothing", "Women's Clothing", "Dresses"),
    ("Free People Jolene Floral Mini Dress Women's S White", "Clothing", "Women's Clothing", "Dresses"),

    # ── CLOTHING > Shoes > Sneakers ───────────────────────────
    ("Nike Air Max 90 Men's Shoes White Size 10", "Clothing", "Shoes", "Sneakers"),
    ("Adidas Ultraboost 22 Running Shoes Men's White Size 11", "Clothing", "Shoes", "Sneakers"),
    ("New Balance 990v6 Men's Grey Size 10.5", "Clothing", "Shoes", "Sneakers"),
    ("Converse Chuck Taylor All Star High Top Black Men's 9", "Clothing", "Shoes", "Sneakers"),
    ("Vans Old Skool Classic Skate Shoe Black/White Men's 10", "Clothing", "Shoes", "Sneakers"),

    # ── HOME & GARDEN > Kitchen > Appliances ──────────────────
    ("KitchenAid Artisan Stand Mixer 5Qt Empire Red KSM150", "Home & Garden", "Kitchen", "Appliances"),
    ("Instant Pot Duo 7-in-1 Electric Pressure Cooker 6Qt", "Home & Garden", "Kitchen", "Appliances"),
    ("Vitamix 5200 Professional Blender 64oz Black", "Home & Garden", "Kitchen", "Appliances"),
    ("Ninja Foodi 9-in-1 Pressure Cooker Air Fryer 6.5Qt", "Home & Garden", "Kitchen", "Appliances"),
    ("Breville Barista Express Espresso Machine Brushed Steel", "Home & Garden", "Kitchen", "Appliances"),
    ("Cuisinart 14-Cup Food Processor Brushed Stainless", "Home & Garden", "Kitchen", "Appliances"),

    # ── HOME & GARDEN > Kitchen > Cookware ────────────────────
    ("All-Clad D3 Stainless Steel 10-Piece Cookware Set", "Home & Garden", "Kitchen", "Cookware"),
    ("Le Creuset Signature Cast Iron Dutch Oven 5.5Qt Flame", "Home & Garden", "Kitchen", "Cookware"),
    ("Caraway Nonstick Ceramic Cookware Set 7-Piece Sage", "Home & Garden", "Kitchen", "Cookware"),
    ("Lodge 12-inch Cast Iron Skillet Pre-Seasoned", "Home & Garden", "Kitchen", "Cookware"),

    # ── HOME & GARDEN > Furniture > Sofas ─────────────────────
    ("IKEA KIVIK 3-Seat Sofa Tibbleby Beige/Grey", "Home & Garden", "Furniture", "Sofas"),
    ("West Elm Harmony Sofa 82in Heathered Crosshatch Slate", "Home & Garden", "Furniture", "Sofas"),
    ("Article Sven Charme Tan Sofa Mid-Century Modern", "Home & Garden", "Furniture", "Sofas"),
    ("Pottery Barn Comfort Roll Arm Slipcovered Sofa Twill White", "Home & Garden", "Furniture", "Sofas"),

    # ── SPORTS > Fitness > Yoga & Pilates ─────────────────────
    ("Lululemon The Reversible Mat 5mm Yoga Mat", "Sports", "Fitness", "Yoga & Pilates"),
    ("Manduka PRO Yoga Mat 6mm Long 85in Black", "Sports", "Fitness", "Yoga & Pilates"),
    ("Gaiam Premium Yoga Mat 6mm Blue Sundial", "Sports", "Fitness", "Yoga & Pilates"),

    # ── SPORTS > Fitness > Weights & Dumbbells ────────────────
    ("Bowflex SelectTech 552 Adjustable Dumbbell Pair", "Sports", "Fitness", "Weights & Dumbbells"),
    ("CAP Barbell 20lb Neoprene Coated Dumbbell Pair", "Sports", "Fitness", "Weights & Dumbbells"),
    ("Amazon Basics Rubber Encased Hex Dumbbell 25lb Single", "Sports", "Fitness", "Weights & Dumbbells"),

    # ── SPORTS > Outdoor > Camping ────────────────────────────
    ("REI Co-op Passage 2 Person Camping Tent", "Sports", "Outdoor", "Camping"),
    ("Coleman Sundome 4-Person Dome Camping Tent Navy", "Sports", "Outdoor", "Camping"),
    ("Big Agnes Copper Spur HV UL2 Ultralight Tent", "Sports", "Outdoor", "Camping"),
    ("MSR Hubba Hubba NX 2-Person Backpacking Tent", "Sports", "Outdoor", "Camping"),

    # ── TOYS > Building Sets > LEGO ───────────────────────────
    ("LEGO Star Wars Millennium Falcon 75192 4016 Pieces", "Toys", "Building Sets", "LEGO"),
    ("LEGO Technic Bugatti Chiron 42083 Building Kit", "Toys", "Building Sets", "LEGO"),
    ("LEGO Architecture Eiffel Tower 10307 10001 Pieces", "Toys", "Building Sets", "LEGO"),
    ("LEGO Harry Potter Hogwarts Castle 71043 6020 Pieces", "Toys", "Building Sets", "LEGO"),

    # ── TOYS > Action Figures > Collectibles ──────────────────
    ("Funko Pop Marvel Spider-Man No Way Home #1158 Vinyl Figure", "Toys", "Action Figures", "Collectibles"),
    ("Hot Wheels 1:18 Scale Die-Cast 1967 Camaro Red", "Toys", "Action Figures", "Collectibles"),
    ("NECA Teenage Mutant Ninja Turtles 1990 Movie 4-Pack", "Toys", "Action Figures", "Collectibles"),
]


def build_dataframe() -> pd.DataFrame:
    """
    Converts the product list into a structured DataFrame.

    Columns:
        title        : product listing title (input to model)
        level_1      : broad category (e.g. "Electronics")
        level_2      : mid category   (e.g. "Audio")
        level_3      : specific category (e.g. "Headphones")
        full_path    : joined path "Electronics > Audio > Headphones"
    """
    rows = []
    for title, l1, l2, l3 in PRODUCTS:
        rows.append({
            "title": title,
            "level_1": l1,
            "level_2": l2,
            "level_3": l3,
            "full_path": f"{l1} > {l2} > {l3}"
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def get_label_maps(df: pd.DataFrame) -> dict:
    """
    Creates integer label mappings for each taxonomy level.

    Why integers? Neural networks work with numbers, not strings.
    So "Electronics" → 0, "Clothing" → 1, etc.

    Returns a dict with:
        l1_to_idx, idx_to_l1
        l2_to_idx, idx_to_l2
        l3_to_idx, idx_to_l3
    """
    l1_classes = sorted(df["level_1"].unique())
    l2_classes = sorted(df["level_2"].unique())
    l3_classes = sorted(df["level_3"].unique())

    return {
        "l1_to_idx": {v: i for i, v in enumerate(l1_classes)},
        "idx_to_l1": {i: v for i, v in enumerate(l1_classes)},
        "l2_to_idx": {v: i for i, v in enumerate(l2_classes)},
        "idx_to_l2": {i: v for i, v in enumerate(l2_classes)},
        "l3_to_idx": {v: i for i, v in enumerate(l3_classes)},
        "idx_to_l3": {i: v for i, v in enumerate(l3_classes)},
        "num_l1": len(l1_classes),
        "num_l2": len(l2_classes),
        "num_l3": len(l3_classes),
    }


def save_dataset(path: str = "data/products.csv"):
    df = build_dataframe()
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} products to {path}")
    print("\nCategory distribution:")
    print(df["full_path"].value_counts().to_string())
    return df


if __name__ == "__main__":
    save_dataset()
