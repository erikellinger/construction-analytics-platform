"""
data_generator.py — Synthetic construction data generator.

Populates the SQLite star schema defined in database.py with realistic
construction project data suitable for development, testing, and demos.

Usage:
    python src/data_generator.py
"""

import calendar
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from faker import Faker
from sqlalchemy import delete, insert
from sqlalchemy.orm import Session

# Allow direct execution from either the project root or src/
sys.path.insert(0, str(Path(__file__).parent))

from database import (
    Base,
    CertificationLevel,
    CostCategory,
    DeliveryStatus,
    DimDate,
    DimEmployee,
    DimEquipment,
    DimMaterial,
    DimProject,
    DimSubcontractor,
    EmployeeStatus,
    EquipmentStatus,
    EquipmentType,
    FactEquipmentUsage,
    FactInspection,
    FactLaborHour,
    FactMaterialDelivery,
    FactProjectCost,
    FactSafetyIncident,
    IncidentType,
    InspectionResult,
    InspectionType,
    MaterialCategory,
    ProjectStatus,
    ProjectType,
    Severity,
    Trade,
    get_engine,
)

# ---------------------------------------------------------------------------
# Reproducibility & globals
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

TODAY = date(2026, 2, 17)
DATA_START = date(2022, 1, 1)
DATA_END = date(2026, 12, 31)

# ---------------------------------------------------------------------------
# CATALOGS
# ---------------------------------------------------------------------------

# (material_name, category, unit, min_unit_cost, max_unit_cost)
MATERIAL_CATALOG = [
    # Lumber
    ('2x4 Stud 8 ft',                        MaterialCategory.LUMBER,      'piece', 5,   9),
    ('2x6 Framing Lumber 8 ft',              MaterialCategory.LUMBER,      'piece', 10,  16),
    ('4x8 OSB Sheathing 7/16"',              MaterialCategory.LUMBER,      'piece', 24,  38),
    ('3/4" Plywood 4x8',                     MaterialCategory.LUMBER,      'piece', 58,  78),
    ('LVL Beam 3.5x11.25 20 ft',             MaterialCategory.LUMBER,      'piece', 190, 260),
    ('Glulam Beam 24 ft',                    MaterialCategory.LUMBER,      'piece', 320, 480),
    ('Treated 6x6 Post 10 ft',               MaterialCategory.LUMBER,      'piece', 55,  85),
    ('Engineered Hardwood Flooring',         MaterialCategory.LUMBER,      'sqft',  6,   14),
    # Concrete
    ('Ready-Mix Concrete 3000 PSI',          MaterialCategory.CONCRETE,    'yard',  145, 185),
    ('Ready-Mix Concrete 4000 PSI',          MaterialCategory.CONCRETE,    'yard',  165, 210),
    ('Concrete Block 8x8x16',               MaterialCategory.CONCRETE,    'piece', 2,   4),
    ('Portland Cement 94 lb bag',           MaterialCategory.CONCRETE,    'piece', 12,  18),
    ('Rebar #4 20 ft',                       MaterialCategory.CONCRETE,    'piece', 14,  22),
    ('Rebar #5 20 ft',                       MaterialCategory.CONCRETE,    'piece', 22,  34),
    ('Wire Mesh 4x8 Panel',                  MaterialCategory.CONCRETE,    'piece', 18,  28),
    ('Shotcrete Mix',                        MaterialCategory.CONCRETE,    'yard',  185, 240),
    # Steel
    ('W8x31 Wide Flange Beam 20 ft',         MaterialCategory.STEEL,       'piece', 380, 520),
    ('W12x45 Wide Flange Beam 30 ft',        MaterialCategory.STEEL,       'piece', 850, 1200),
    ('HSS Tube Steel 4x4 20 ft',             MaterialCategory.STEEL,       'piece', 120, 180),
    ('Steel Plate 3/8" 4x8',                 MaterialCategory.STEEL,       'piece', 280, 420),
    ('Structural Steel Angle 20 ft',         MaterialCategory.STEEL,       'piece', 90,  140),
    ('Steel Decking 26 ga',                  MaterialCategory.STEEL,       'sqft',  3,   6),
    ('Corrugated Metal Roofing',             MaterialCategory.STEEL,       'sqft',  4,   8),
    ('Light-gauge Steel Stud 10 ft',         MaterialCategory.STEEL,       'piece', 8,   14),
    # Electrical
    ('12/2 Romex 250 ft',                    MaterialCategory.ELECTRICAL,  'roll',  65,  95),
    ('10/3 Romex 250 ft',                    MaterialCategory.ELECTRICAL,  'roll',  110, 155),
    ('1" EMT Conduit 10 ft',                 MaterialCategory.ELECTRICAL,  'piece', 12,  20),
    ('2" PVC Conduit 10 ft',                 MaterialCategory.ELECTRICAL,  'piece', 18,  28),
    ('200A Main Panel 24-circuit',           MaterialCategory.ELECTRICAL,  'piece', 280, 420),
    ('20A Single-Pole Breaker',              MaterialCategory.ELECTRICAL,  'piece', 8,   18),
    ('Duplex Outlets 10-pack',               MaterialCategory.ELECTRICAL,  'piece', 24,  40),
    ('4-ft LED Shop Light 2-pack',           MaterialCategory.ELECTRICAL,  'piece', 55,  90),
    ('Junction Box 4x4',                     MaterialCategory.ELECTRICAL,  'piece', 5,   10),
    # Plumbing
    ('3/4" Copper Pipe 10 ft',               MaterialCategory.PLUMBING,    'piece', 28,  45),
    ('1/2" Copper Pipe 10 ft',               MaterialCategory.PLUMBING,    'piece', 18,  30),
    ('4" PVC DWV Pipe 10 ft',                MaterialCategory.PLUMBING,    'piece', 20,  34),
    ('3" ABS Pipe 10 ft',                    MaterialCategory.PLUMBING,    'piece', 22,  36),
    ('3/4" Ball Valve',                      MaterialCategory.PLUMBING,    'piece', 18,  35),
    ('50-gal Water Heater',                  MaterialCategory.PLUMBING,    'piece', 550, 900),
    ('Standard Toilet',                      MaterialCategory.PLUMBING,    'piece', 180, 400),
    ('Stainless Kitchen Sink',               MaterialCategory.PLUMBING,    'piece', 220, 450),
    ('Sump Pump',                            MaterialCategory.PLUMBING,    'piece', 180, 350),
    # Insulation
    ('Fiberglass Batt R-19 40 sqft',         MaterialCategory.INSULATION,  'piece', 28,  45),
    ('Fiberglass Batt R-38 40 sqft',         MaterialCategory.INSULATION,  'piece', 48,  72),
    ('Rigid Foam 2" 4x8',                    MaterialCategory.INSULATION,  'piece', 24,  38),
    ('Spray Foam 600 bdft Kit',              MaterialCategory.INSULATION,  'piece', 380, 520),
    ('Blown Cellulose 25 lb Bag',            MaterialCategory.INSULATION,  'piece', 18,  30),
    ('Mineral Wool Batt R-15',               MaterialCategory.INSULATION,  'piece', 34,  52),
    # Finishes
    ('Drywall 1/2" 4x8',                     MaterialCategory.FINISHES,    'piece', 14,  22),
    ('Drywall 5/8" Fire-Rated 4x8',          MaterialCategory.FINISHES,    'piece', 18,  28),
    ('Joint Compound 5-gal',                 MaterialCategory.FINISHES,    'piece', 22,  34),
    ('Exterior Paint 5-gal',                 MaterialCategory.FINISHES,    'piece', 90,  145),
    ('Interior Paint 5-gal',                 MaterialCategory.FINISHES,    'piece', 75,  125),
    ('Ceramic Tile 12x12 Case',              MaterialCategory.FINISHES,    'piece', 35,  75),
    ('Porcelain Tile 24x24 Case',            MaterialCategory.FINISHES,    'piece', 85,  165),
    ('Laminate Flooring 20 sqft',            MaterialCategory.FINISHES,    'piece', 55,  95),
    ('Carpet',                               MaterialCategory.FINISHES,    'yard',  18,  45),
    ('Baseboard Trim 16 ft',                 MaterialCategory.FINISHES,    'piece', 22,  40),
    ('Pre-hung Interior Door',               MaterialCategory.FINISHES,    'piece', 180, 320),
    ('Fiberglass Exterior Door',             MaterialCategory.FINISHES,    'piece', 480, 850),
    ('Double-hung Window 3x4',               MaterialCategory.FINISHES,    'piece', 280, 550),
    # Other
    ('Asphalt Shingles per Square',          MaterialCategory.OTHER,       'piece', 85,  145),
    ('30 lb Roofing Felt 4-sq Roll',         MaterialCategory.OTHER,       'piece', 38,  58),
    ('House Wrap 9x100 ft',                  MaterialCategory.OTHER,       'piece', 140, 200),
    ('#57 Stone Gravel',                     MaterialCategory.OTHER,       'ton',   32,  55),
    ('Washed Sand',                          MaterialCategory.OTHER,       'ton',   25,  42),
    ('Topsoil',                              MaterialCategory.OTHER,       'yard',  28,  48),
    ('Concrete Anchor Bolts 50-pk',          MaterialCategory.OTHER,       'piece', 28,  45),
    ('Exterior Caulk Case of 12',            MaterialCategory.OTHER,       'piece', 55,  85),
    ('Safety Orange Mesh Netting 100 ft',    MaterialCategory.OTHER,       'piece', 32,  55),
    ('Landscape Fabric 4x250 ft',            MaterialCategory.OTHER,       'roll',  55,  85),
]

# (equipment_name, type, min_purchase, max_purchase, min_hourly, max_hourly)
EQUIPMENT_CATALOG = [
    ('CAT 320 Hydraulic Excavator',          EquipmentType.EXCAVATOR,  185_000, 360_000, 90,  145),
    ('Komatsu PC210 Excavator',              EquipmentType.EXCAVATOR,  175_000, 340_000, 85,  140),
    ('Volvo EC220 Excavator',                EquipmentType.EXCAVATOR,  180_000, 350_000, 88,  142),
    ('CAT 308 Mini Excavator',               EquipmentType.EXCAVATOR,   65_000, 110_000, 50,   85),
    ('John Deere 75G Excavator',             EquipmentType.EXCAVATOR,   70_000, 115_000, 52,   88),
    ('Liebherr LTM 1100 Mobile Crane',       EquipmentType.CRANE,      850_000,1_800_000,300,  600),
    ('Grove RT760E Rough Terrain Crane',     EquipmentType.CRANE,      450_000,  900_000,220,  420),
    ('Manitowoc 18000 Crawler Crane',        EquipmentType.CRANE,    1_200_000,2_500_000,500,  950),
    ('Tadano GR-1000XL Crane',               EquipmentType.CRANE,      700_000,1_400_000,280,  550),
    ('CAT D6T Bulldozer',                    EquipmentType.BULLDOZER,  250_000, 450_000,110,  180),
    ('Komatsu D51 Bulldozer',                EquipmentType.BULLDOZER,  230_000, 420_000,105,  170),
    ('CAT D8T Large Bulldozer',              EquipmentType.BULLDOZER,  450_000, 750_000,160,  260),
    ('John Deere 700K Dozer',                EquipmentType.BULLDOZER,  210_000, 390_000,100,  165),
    ('Mack Granite Dump Truck',              EquipmentType.TRUCK,      120_000, 200_000, 55,   95),
    ('Peterbilt 389 Flatbed',                EquipmentType.TRUCK,      130_000, 210_000, 60,  100),
    ('Ford F-750 Service Truck',             EquipmentType.TRUCK,       70_000, 120_000, 35,   65),
    ('CAT CT660 Vocational Truck',           EquipmentType.TRUCK,      110_000, 185_000, 50,   88),
    ('Kenworth T680 Haul Truck',             EquipmentType.TRUCK,      125_000, 205_000, 58,   96),
    ('Freightliner Concrete Mixer',          EquipmentType.TRUCK,      165_000, 250_000, 75,  125),
    ('Toyota 8FGU25 Forklift',               EquipmentType.FORKLIFT,    28_000,  52_000, 22,   40),
    ('CAT C5000 Forklift',                   EquipmentType.FORKLIFT,    30_000,  58_000, 24,   42),
    ('JLG G10-55A Telehandler',              EquipmentType.FORKLIFT,    85_000, 155_000, 55,   95),
    ('Manitou MRT 2540 Telehandler',         EquipmentType.FORKLIFT,    90_000, 165_000, 58,   98),
    ('CAT DE220E0 Generator 220kW',          EquipmentType.GENERATOR,   38_000,  75_000, 18,   35),
    ('Generac SG200 Generator 200kW',        EquipmentType.GENERATOR,   32_000,  65_000, 16,   32),
    ('Kohler 150REOZVB Generator',           EquipmentType.GENERATOR,   28_000,  55_000, 14,   28),
    ('CAT CB10 Asphalt Compactor',           EquipmentType.COMPACTOR,   88_000, 160_000, 55,   95),
    ('Wacker Neuson VP1550A Plate Compactor',EquipmentType.COMPACTOR,    4_500,   8_500, 20,   38),
    ('Bomag BW 213 DH Drum Roller',          EquipmentType.COMPACTOR,   95_000, 175_000, 60,  100),
    ('Dynapac CA2500 Soil Compactor',        EquipmentType.COMPACTOR,  105_000, 190_000, 65,  110),
]

# Construction phases: ordered list from foundation to finish
PHASES = [
    {
        "name": "foundation",
        "pct_start": 0.00, "pct_end": 0.20,
        "trades": [Trade.CONCRETE, Trade.GENERAL],
        "equip_types": [EquipmentType.EXCAVATOR, EquipmentType.BULLDOZER,
                        EquipmentType.TRUCK, EquipmentType.COMPACTOR],
        "mat_cats": [MaterialCategory.CONCRETE, MaterialCategory.STEEL, MaterialCategory.OTHER],
        "cost_weight": 0.15,
    },
    {
        "name": "framing",
        "pct_start": 0.20, "pct_end": 0.45,
        "trades": [Trade.FRAMING, Trade.GENERAL],
        "equip_types": [EquipmentType.CRANE, EquipmentType.FORKLIFT, EquipmentType.TRUCK],
        "mat_cats": [MaterialCategory.LUMBER, MaterialCategory.STEEL, MaterialCategory.CONCRETE],
        "cost_weight": 0.22,
    },
    {
        "name": "systems",
        "pct_start": 0.45, "pct_end": 0.75,
        "trades": [Trade.ELECTRICAL, Trade.PLUMBING, Trade.HVAC],
        "equip_types": [EquipmentType.FORKLIFT, EquipmentType.GENERATOR, EquipmentType.TRUCK],
        "mat_cats": [MaterialCategory.ELECTRICAL, MaterialCategory.PLUMBING,
                     MaterialCategory.INSULATION],
        "cost_weight": 0.38,
    },
    {
        "name": "finishing",
        "pct_start": 0.75, "pct_end": 1.00,
        "trades": [Trade.PAINTING, Trade.ELECTRICAL, Trade.GENERAL],
        "equip_types": [EquipmentType.FORKLIFT, EquipmentType.TRUCK, EquipmentType.GENERATOR],
        "mat_cats": [MaterialCategory.FINISHES, MaterialCategory.OTHER],
        "cost_weight": 0.25,
    },
]

LOCATIONS = [
    "St. Louis, MO", "Kansas City, MO", "Springfield, MO", "Jefferson City, MO",
    "Columbia, MO", "Joplin, MO", "St. Charles, MO", "O'Fallon, MO",
    "Chesterfield, MO", "Cape Girardeau, MO", "Kirkwood, MO", "Clayton, MO",
    "Ballwin, MO", "Florissant, MO", "Hazelwood, MO", "St. Peters, MO",
    "Wentzville, MO", "Lee's Summit, MO", "Independence, MO", "Blue Springs, MO",
]

CLIENT_NAMES = [
    "Midwest Development Group", "Gateway Realty Partners", "Arch Capital Investments",
    "Missouri State DOT", "St. Louis County Public Works", "KC Metro Authority",
    "Heartland Commercial Properties", "Show-Me Infrastructure LLC",
    "River City Development", "Ozark Construction Partners",
    "Cardinal Real Estate Group", "Boone County Developers",
    "Prairie Land Investments", "Confluence Group LLC", "Highland Capital Partners",
    "Crestwood Holdings", "Barton Creek Development", "Eagle Ridge Properties",
    "Summit Commercial Group", "Trailhead Real Estate LLC",
]

SUPPLIERS = [
    "ABC Supply Co.", "Home Depot Pro", "Menards Contractor Supply", "84 Lumber",
    "Fastenal Industrial", "Ferguson Enterprises", "Graybar Electric",
    "Consolidated Electrical Distributors", "Vulcan Materials", "Martin Marietta Aggregates",
    "USG Corporation", "Holcim US", "Nucor Steel", "Commercial Metals Company",
    "Wesco International", "Anixter Wire & Cable", "Hajoca Plumbing Supply",
    "Industrial Distribution Group", "Builders FirstSource", "ProBuild Holdings",
]

RES_PREFIXES = [
    "Oakwood", "Ridgeview", "Creekside", "Westbrook", "Lakeside", "Meadowland",
    "Hillcrest", "Riverside", "Pinecrest", "Sunrise", "Stonegate", "Willow Run",
    "Foxwood", "Briarhaven", "Clearwater", "Evergreen", "Millbrook", "Cedarhill",
]
RES_SUFFIXES = [
    "Residences", "Townhomes", "Commons", "Estates", "Villas",
    "Heights", "Park", "Crossing", "Ridge", "Place", "Pointe", "Landing",
]
COM_PREFIXES = [
    "Gateway", "Midtown", "Westport", "Northside", "Commerce", "Keystone",
    "Summit", "Premier", "Pinnacle", "Central", "Landmark", "Metro",
]
COM_SUFFIXES = [
    "Business Center", "Corporate Campus", "Plaza", "Medical Center",
    "Office Park", "Professional Center", "Trade Center", "Hub", "Complex",
]
INFRA_TEMPLATES = [
    "{city} Bridge Reconstruction",
    "{city} Highway Interchange Upgrade",
    "Route {num} Overpass Replacement",
    "I-{num} Corridor Expansion",
    "{city} Water Treatment Plant Expansion",
    "{city} Wastewater Infrastructure Upgrade",
    "{city} Utility Corridor Phase {num}",
    "{city} Flood Control & Drainage Project",
    "{city} Road Widening Project",
]

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


def add_months(d: date, n: int) -> date:
    """Advance date d by n months, clamping to end-of-month."""
    m = d.month - 1 + n
    year = d.year + m // 12
    month = m % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _us_holidays(year: int) -> set:
    def nth_weekday(y, mo, wd, n):
        first = date(y, mo, 1)
        offset = (wd - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))

    def last_weekday(y, mo, wd):
        last = date(y, mo, calendar.monthrange(y, mo)[1])
        return last - timedelta(days=(last.weekday() - wd) % 7)

    return {
        date(year, 1, 1),               # New Year's Day
        nth_weekday(year, 1, 0, 3),     # MLK Day (3rd Mon Jan)
        nth_weekday(year, 2, 0, 3),     # Presidents' Day (3rd Mon Feb)
        last_weekday(year, 5, 0),       # Memorial Day (last Mon May)
        date(year, 7, 4),               # Independence Day
        nth_weekday(year, 9, 0, 1),     # Labor Day (1st Mon Sep)
        nth_weekday(year, 11, 3, 4),    # Thanksgiving (4th Thu Nov)
        date(year, 12, 25),             # Christmas
    }


_holiday_cache: dict = {}


def is_holiday(d: date) -> bool:
    if d.year not in _holiday_cache:
        _holiday_cache[d.year] = _us_holidays(d.year)
    return d in _holiday_cache[d.year]


def phase_for_date(proj_start: date, proj_end: date, current: date) -> dict:
    """Return the phase dict that covers `current` within the project timeline."""
    total = max((proj_end - proj_start).days, 1)
    progress = max(0.0, min((current - proj_start).days / total, 0.9999))
    for phase in PHASES:
        if phase["pct_start"] <= progress < phase["pct_end"]:
            return phase
    return PHASES[-1]


def bulk_insert(session: Session, model, rows: list, batch_size: int = 2_000) -> None:
    """Core-style bulk insert — much faster than ORM add_all for large volumes."""
    for i in range(0, len(rows), batch_size):
        session.execute(insert(model), rows[i : i + batch_size])
    session.commit()


# ---------------------------------------------------------------------------
# DIMENSION GENERATORS
# ---------------------------------------------------------------------------


def generate_dim_dates(session: Session) -> None:
    """Fill dim_dates for every calendar day from DATA_START to DATA_END."""
    print("  Generating dim_dates ...", end=" ", flush=True)
    if session.get(DimDate, DATA_START):
        print("already populated, skipping.")
        return

    rows, current = [], DATA_START
    while current <= DATA_END:
        rows.append({
            "date":        current,
            "year":        current.year,
            "quarter":     (current.month - 1) // 3 + 1,
            "month":       current.month,
            "month_name":  current.strftime("%B"),
            "week":        current.isocalendar()[1],
            "day_of_week": current.weekday(),   # 0=Mon, 6=Sun
            "is_weekend":  current.weekday() >= 5,
            "is_holiday":  is_holiday(current),
        })
        current += timedelta(days=1)

    bulk_insert(session, DimDate, rows)
    print(f"{len(rows):,} rows inserted.")


def generate_projects(session: Session, n: int = 57) -> list:
    if session.query(DimProject).count() > 0:
        print("  dim_projects already populated, loading.")
        return session.query(DimProject).all()

    print(f"  Generating {n} projects ...", end=" ", flush=True)

    type_pool = (
        [ProjectType.RESIDENTIAL] * 26
        + [ProjectType.COMMERCIAL] * 21
        + [ProjectType.INFRASTRUCTURE] * 10
    )
    random.shuffle(type_pool)
    type_pool = type_pool[:n]

    objs = []
    for ptype in type_pool:
        duration_months = random.randint(3, 18)

        if ptype == ProjectType.RESIDENTIAL:
            contract_value = random.uniform(600_000, 10_000_000)
            sqft = random.uniform(2_000, 18_000)
        elif ptype == ProjectType.COMMERCIAL:
            contract_value = random.uniform(2_000_000, 65_000_000)
            sqft = random.uniform(10_000, 250_000)
        else:
            contract_value = random.uniform(5_000_000, 175_000_000)
            sqft = None

        # Scatter planned starts across 2022-01-01 to 2025-09-01
        days_spread = (date(2025, 9, 1) - DATA_START).days
        planned_start = DATA_START + timedelta(days=random.randint(0, days_spread))
        planned_end = add_months(planned_start, duration_months)

        # Status & actual dates
        if planned_start > TODAY:
            status = ProjectStatus.PLANNING
            actual_start = actual_end = None
        elif planned_end < date(2024, 6, 1):
            # Old enough to definitely be done
            status = ProjectStatus.COMPLETED
            slip = random.uniform(0.90, 1.30)
            actual_start = planned_start + timedelta(days=random.randint(-5, 20))
            actual_end = add_months(actual_start, int(duration_months * slip))
        elif planned_end < TODAY:
            if random.random() < 0.82:
                status = ProjectStatus.COMPLETED
                slip = random.uniform(0.92, 1.28)
                actual_start = planned_start + timedelta(days=random.randint(-5, 20))
                actual_end = add_months(actual_start, int(duration_months * slip))
            else:
                status = ProjectStatus.ON_HOLD
                actual_start = planned_start
                actual_end = None
        else:
            status = random.choices(
                [ProjectStatus.ACTIVE, ProjectStatus.ON_HOLD], weights=[92, 8]
            )[0]
            actual_start = planned_start + timedelta(days=random.randint(-5, 15))
            actual_end = None

        # Build name
        if ptype == ProjectType.RESIDENTIAL:
            name = f"{random.choice(RES_PREFIXES)} {random.choice(RES_SUFFIXES)}"
        elif ptype == ProjectType.COMMERCIAL:
            name = f"{random.choice(COM_PREFIXES)} {random.choice(COM_SUFFIXES)}"
        else:
            tmpl = random.choice(INFRA_TEMPLATES)
            city = random.choice(LOCATIONS).split(",")[0]
            name = tmpl.format(city=city, num=random.randint(47, 270))

        # Clamp actual_end to DATA_END so fact date_ids stay in dim_dates
        if actual_end and actual_end > DATA_END:
            actual_end = DATA_END

        objs.append(DimProject(
            project_name=name,
            project_type=ptype.value,
            contract_value=round(contract_value, 2),
            planned_start_date=planned_start,
            planned_end_date=planned_end,
            actual_start_date=actual_start,
            actual_end_date=actual_end,
            status=status.value,
            client_name=random.choice(CLIENT_NAMES),
            location=random.choice(LOCATIONS),
            square_footage=round(sqft) if sqft else None,
        ))

    session.add_all(objs)
    session.commit()
    print(f"{len(objs)} rows inserted.")
    return session.query(DimProject).all()


def generate_subcontractors(session: Session, n: int = 210) -> list:
    if session.query(DimSubcontractor).count() > 0:
        print("  dim_subcontractors already populated, loading.")
        return session.query(DimSubcontractor).all()

    print(f"  Generating {n} subcontractors ...", end=" ", flush=True)

    # Weighted trade distribution matching real industry proportions
    trade_weights = {
        Trade.ELECTRICAL: 18, Trade.PLUMBING: 16, Trade.HVAC: 14,
        Trade.FRAMING: 14, Trade.CONCRETE: 13, Trade.ROOFING: 10,
        Trade.PAINTING: 10, Trade.LANDSCAPING: 8, Trade.GENERAL: 7,
    }
    trade_pool = [t for t, w in trade_weights.items() for _ in range(w)]

    company_suffixes = ["Contracting", "Construction", "Services", "Group", "Inc.", "LLC",
                        "& Sons", "Co.", "Mechanical", "Electric"]
    objs = []
    for _ in range(n):
        objs.append(DimSubcontractor(
            company_name=f"{fake.last_name()} {random.choice(company_suffixes)}",
            trade=random.choice(trade_pool).value,
            contact_person=fake.name(),
            phone=fake.numerify("(###) ###-####"),
            email=fake.company_email(),
            rating=round(random.uniform(2.5, 5.0), 1),
            years_in_business=random.randint(2, 38),
        ))

    session.add_all(objs)
    session.commit()
    print(f"{len(objs)} rows inserted.")
    return session.query(DimSubcontractor).all()


def generate_equipment(session: Session, n: int = 105) -> list:
    if session.query(DimEquipment).count() > 0:
        print("  dim_equipment already populated, loading.")
        return session.query(DimEquipment).all()

    print(f"  Generating {n} pieces of equipment ...", end=" ", flush=True)
    objs = []
    for i in range(n):
        name, etype, min_cost, max_cost, min_rate, max_rate = EQUIPMENT_CATALOG[i % len(EQUIPMENT_CATALOG)]
        # Add a unit number suffix for duplicated types
        suffix = f" #{i // len(EQUIPMENT_CATALOG) + 1}" if i >= len(EQUIPMENT_CATALOG) else ""
        purchase_year = random.randint(2012, 2024)
        age = 2025 - purchase_year

        # Older machines more likely to be retired or in maintenance
        if age > 10 and random.random() < 0.12:
            status = EquipmentStatus.RETIRED
        elif random.random() < 0.08:
            status = EquipmentStatus.MAINTENANCE
        else:
            status = EquipmentStatus.ACTIVE

        objs.append(DimEquipment(
            equipment_name=name + suffix,
            equipment_type=etype.value,
            purchase_date=date(purchase_year, random.randint(1, 12), random.randint(1, 28)),
            purchase_cost=round(random.uniform(min_cost, max_cost), 2),
            hourly_rate=round(random.uniform(min_rate, max_rate), 2),
            status=status.value,
        ))

    session.add_all(objs)
    session.commit()
    print(f"{len(objs)} rows inserted.")
    return session.query(DimEquipment).all()


def generate_employees(session: Session, n: int = 525) -> list:
    if session.query(DimEmployee).count() > 0:
        print("  dim_employees already populated, loading.")
        return session.query(DimEmployee).all()

    print(f"  Generating {n} employees ...", end=" ", flush=True)

    trade_weights = {
        Trade.GENERAL: 22, Trade.FRAMING: 14, Trade.ELECTRICAL: 14,
        Trade.PLUMBING: 12, Trade.HVAC: 10, Trade.CONCRETE: 10,
        Trade.ROOFING: 8,  Trade.PAINTING: 8, Trade.LANDSCAPING: 2,
    }
    # Hourly rate ranges keyed by (trade, cert_level)
    rate_table = {
        (Trade.GENERAL,     CertificationLevel.APPRENTICE): (17, 23),
        (Trade.GENERAL,     CertificationLevel.JOURNEYMAN): (24, 34),
        (Trade.GENERAL,     CertificationLevel.MASTER):     (35, 50),
        (Trade.FRAMING,     CertificationLevel.APPRENTICE): (19, 25),
        (Trade.FRAMING,     CertificationLevel.JOURNEYMAN): (28, 38),
        (Trade.FRAMING,     CertificationLevel.MASTER):     (40, 55),
        (Trade.ELECTRICAL,  CertificationLevel.APPRENTICE): (20, 27),
        (Trade.ELECTRICAL,  CertificationLevel.JOURNEYMAN): (32, 48),
        (Trade.ELECTRICAL,  CertificationLevel.MASTER):     (52, 78),
        (Trade.PLUMBING,    CertificationLevel.APPRENTICE): (20, 26),
        (Trade.PLUMBING,    CertificationLevel.JOURNEYMAN): (30, 46),
        (Trade.PLUMBING,    CertificationLevel.MASTER):     (50, 75),
        (Trade.HVAC,        CertificationLevel.APPRENTICE): (19, 26),
        (Trade.HVAC,        CertificationLevel.JOURNEYMAN): (30, 46),
        (Trade.HVAC,        CertificationLevel.MASTER):     (50, 72),
        (Trade.CONCRETE,    CertificationLevel.APPRENTICE): (18, 24),
        (Trade.CONCRETE,    CertificationLevel.JOURNEYMAN): (26, 38),
        (Trade.CONCRETE,    CertificationLevel.MASTER):     (38, 56),
        (Trade.ROOFING,     CertificationLevel.APPRENTICE): (18, 24),
        (Trade.ROOFING,     CertificationLevel.JOURNEYMAN): (26, 36),
        (Trade.ROOFING,     CertificationLevel.MASTER):     (38, 55),
        (Trade.PAINTING,    CertificationLevel.APPRENTICE): (16, 22),
        (Trade.PAINTING,    CertificationLevel.JOURNEYMAN): (24, 34),
        (Trade.PAINTING,    CertificationLevel.MASTER):     (36, 50),
        (Trade.LANDSCAPING, CertificationLevel.APPRENTICE): (15, 20),
        (Trade.LANDSCAPING, CertificationLevel.JOURNEYMAN): (22, 30),
        (Trade.LANDSCAPING, CertificationLevel.MASTER):     (32, 48),
    }

    trade_pool = [t for t, w in trade_weights.items() for _ in range(w)]
    cert_pool  = (
        [CertificationLevel.APPRENTICE] * 40
        + [CertificationLevel.JOURNEYMAN] * 45
        + [CertificationLevel.MASTER] * 15
    )

    objs = []
    for _ in range(n):
        trade = random.choice(trade_pool)
        cert  = random.choice(cert_pool)
        lo, hi = rate_table[(trade, cert)]
        hire_year = random.randint(2005, 2024)

        objs.append(DimEmployee(
            employee_name=fake.name(),
            trade=trade.value,
            hourly_rate=round(random.uniform(lo, hi), 2),
            hire_date=date(hire_year, random.randint(1, 12), random.randint(1, 28)),
            certification_level=cert.value,
            status=EmployeeStatus.INACTIVE.value if random.random() < 0.07
                   else EmployeeStatus.ACTIVE.value,
        ))

    session.add_all(objs)
    session.commit()
    print(f"{len(objs)} rows inserted.")
    return session.query(DimEmployee).all()


def generate_materials(session: Session) -> list:
    if session.query(DimMaterial).count() > 0:
        print("  dim_materials already populated, loading.")
        return session.query(DimMaterial).all()

    n = len(MATERIAL_CATALOG)
    print(f"  Generating {n} materials ...", end=" ", flush=True)

    objs = []
    for name, category, unit, lo, hi in MATERIAL_CATALOG:
        objs.append(DimMaterial(
            material_name=name,
            category=category.value,
            unit=unit,
            unit_cost=round(random.uniform(lo, hi), 2),
            supplier=random.choice(SUPPLIERS),
        ))

    session.add_all(objs)
    session.commit()
    print(f"{len(objs)} rows inserted.")
    return session.query(DimMaterial).all()


# ---------------------------------------------------------------------------
# FACT GENERATORS  (one project at a time)
# ---------------------------------------------------------------------------


def _effective_range(p: DimProject) -> Optional[tuple]:
    """Return (data_start, data_end) for a project, or None if not yet started."""
    start = p.actual_start_date
    if not start:
        return None
    # Use actual end for completed projects; cap active projects at TODAY
    end = p.actual_end_date or min(TODAY, p.planned_end_date or TODAY)
    end = min(end, DATA_END)   # never exceed dim_dates range
    if end <= start:
        return None
    return start, end


def _phase_end_ref(p: DimProject) -> date:
    """The full planned/actual end date, used for phase-progress calculations."""
    return p.actual_end_date or p.planned_end_date or add_months(p.actual_start_date, 12)


def _gen_costs(project: DimProject, start: date, end: date,
               subcontractors: list) -> list:
    """Weekly cost entries spread realistically across phases and categories."""
    cost_basis = project.contract_value * random.uniform(0.68, 0.88)
    budget_factor = random.uniform(0.88, 1.32)
    actual_total = cost_basis * budget_factor
    budget_total = cost_basis

    phase_ref_end = _phase_end_ref(project)

    # Collect all Monday dates in the project's data window
    weekly_dates = []
    d = start
    while d <= end:
        if d.weekday() == 0:
            weekly_dates.append(d)
        d += timedelta(days=1)
    if not weekly_dates:
        weekly_dates = [start]

    # Pre-count weeks per phase so we can distribute budget evenly
    phase_week_counts: dict = {}
    for wd in weekly_dates:
        ph = phase_for_date(start, phase_ref_end, wd)["name"]
        phase_week_counts[ph] = phase_week_counts.get(ph, 0) + 1

    cat_splits = {
        CostCategory.LABOR:         0.35,
        CostCategory.MATERIALS:     0.28,
        CostCategory.EQUIPMENT:     0.12,
        CostCategory.SUBCONTRACTOR: 0.20,
        CostCategory.OTHER:         0.05,
    }

    # Pick 2-4 subcontractors associated with this project
    project_subs = random.sample(subcontractors, min(4, len(subcontractors)))

    rows = []
    for wd in weekly_dates:
        phase = phase_for_date(start, phase_ref_end, wd)
        weeks_in_phase = max(1, phase_week_counts.get(phase["name"], 1))

        for category, cat_split in cat_splits.items():
            # Apportion this week's share of that category's phase budget
            base_actual = (actual_total * cat_split * phase["cost_weight"]) / weeks_in_phase
            base_budget = (budget_total * cat_split * phase["cost_weight"]) / weeks_in_phase
            amount  = round(base_actual * random.uniform(0.75, 1.25), 2)
            budgeted = round(base_budget * random.uniform(0.95, 1.05), 2)

            sub_id = None
            if category == CostCategory.SUBCONTRACTOR and project_subs:
                sub_id = random.choice(project_subs).id

            rows.append({
                "project_id":       project.id,
                "date_id":          wd,
                "cost_category":    category.value,
                "subcontractor_id": sub_id,
                "amount":           amount,
                "budgeted_amount":  budgeted,
                "notes":            None,
            })

    return rows


def _gen_labor(project: DimProject, start: date, end: date,
               employees: list) -> list:
    """Daily labor hour entries for a project crew, weekdays only."""
    active_emps = [e for e in employees if e.status == EmployeeStatus.ACTIVE.value]
    crew = random.sample(active_emps, min(random.randint(10, 20), len(active_emps)))
    phase_ref_end = _phase_end_ref(project)

    rows = []
    current = start
    while current <= end:
        # Skip Sundays; allow rare Saturday crunch (5%)
        if current.weekday() == 6:
            current += timedelta(days=1)
            continue
        if current.weekday() == 5 and random.random() > 0.05:
            current += timedelta(days=1)
            continue
        if is_holiday(current) and random.random() > 0.04:
            current += timedelta(days=1)
            continue

        phase = phase_for_date(start, phase_ref_end, current)
        phase_trade_vals = {t.value for t in phase["trades"]}

        # Preferred trades show up at 92%, others at 45%
        for emp in crew:
            if emp.trade in phase_trade_vals:
                if random.random() > 0.92:
                    continue
            else:
                if random.random() > 0.45:
                    continue
            rows.append({
                "project_id":       project.id,
                "employee_id":      emp.id,
                "date_id":          current,
                "hours_worked":     round(random.uniform(6.0, 10.0), 1),
                "hourly_rate":      emp.hourly_rate,
                "task_description": None,
            })

        current += timedelta(days=1)

    return rows


def _gen_equipment_usage(project: DimProject, start: date, end: date,
                         fleet: list, employees: list) -> list:
    """Daily equipment usage logs, phase-weighted by equipment type."""
    active_fleet = [e for e in fleet if e.status == EquipmentStatus.ACTIVE.value]
    active_ops   = [e for e in employees if e.status == EmployeeStatus.ACTIVE.value]
    if not active_fleet or not active_ops:
        return []

    assigned = random.sample(active_fleet, min(random.randint(3, 7), len(active_fleet)))
    phase_ref_end = _phase_end_ref(project)

    rows = []
    current = start
    while current <= end:
        if current.weekday() >= 5 or is_holiday(current):
            current += timedelta(days=1)
            continue

        phase = phase_for_date(start, phase_ref_end, current)
        phase_equip_vals = {t.value for t in phase["equip_types"]}

        for equip in assigned:
            relevant = equip.equipment_type in phase_equip_vals
            if random.random() > (0.78 if relevant else 0.22):
                continue
            if random.random() < 0.05:   # maintenance day
                continue

            hours = round(random.uniform(2.0, 10.0), 1)
            rows.append({
                "project_id":       project.id,
                "equipment_id":     equip.id,
                "date_id":          current,
                "hours_used":       hours,
                "fuel_cost":        round(hours * random.uniform(12, 35), 2) if relevant else None,
                "maintenance_cost": round(random.uniform(50, 800), 2) if random.random() < 0.04 else None,
                "operator_id":      random.choice(active_ops).id,
            })

        current += timedelta(days=1)

    return rows


def _gen_material_deliveries(project: DimProject, start: date, end: date,
                              materials: list) -> list:
    """Staggered material deliveries matching the current construction phase."""
    phase_ref_end = _phase_end_ref(project)

    # Build a list of delivery dates (roughly every 7-10 working days)
    delivery_dates = []
    d = start + timedelta(days=random.randint(2, 7))
    while d <= end:
        if d.weekday() < 5:
            delivery_dates.append(d)
        d += timedelta(days=random.randint(5, 12))

    rows = []
    for dd in delivery_dates:
        phase = phase_for_date(start, phase_ref_end, dd)
        phase_cat_vals = {c.value for c in phase["mat_cats"]}

        relevant = [m for m in materials if m.category in phase_cat_vals]
        batch = random.sample(relevant or materials, min(random.randint(1, 4), len(relevant or materials)))

        for mat in batch:
            if mat.unit in ("piece", "roll"):
                qty = round(random.uniform(10, 500))
            elif mat.unit == "sqft":
                qty = round(random.uniform(200, 5_000))
            elif mat.unit in ("ton", "yard"):
                qty = round(random.uniform(2, 80), 1)
            else:
                qty = round(random.uniform(5, 200))

            rows.append({
                "project_id":      project.id,
                "material_id":     mat.id,
                "date_id":         dd,
                "quantity":        qty,
                "total_cost":      round(qty * mat.unit_cost * random.uniform(0.95, 1.08), 2),
                "delivery_status": (DeliveryStatus.ON_TIME if random.random() < 0.88
                                    else DeliveryStatus.DELAYED).value,
            })

    return rows


def _gen_safety_incidents(project: DimProject, start: date, end: date) -> list:
    """Rare safety incidents — roughly 35% of active projects see at least one."""
    if random.random() > 0.35:
        return []

    n = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
    total_days = max((end - start).days, 1)
    rows = []

    for _ in range(n):
        # Incidents cluster toward the earlier, higher-risk phases
        offset = min(int(abs(random.gauss(0.25, 0.20)) * total_days), total_days - 1)
        d = start + timedelta(days=offset)
        while d.weekday() >= 5 and d <= end:
            d += timedelta(days=1)
        if d > end:
            continue

        severity = random.choices(
            [Severity.MINOR, Severity.MODERATE, Severity.SEVERE],
            weights=[70, 25, 5],
        )[0]
        itype = random.choices(
            [IncidentType.NEAR_MISS, IncidentType.INJURY, IncidentType.VIOLATION],
            weights=[45, 35, 20],
        )[0]

        days_lost = 0
        if itype == IncidentType.INJURY:
            days_lost = {
                Severity.MINOR:    random.randint(0, 2),
                Severity.MODERATE: random.randint(1, 10),
                Severity.SEVERE:   random.randint(5, 30),
            }[severity]

        rows.append({
            "project_id":   project.id,
            "date_id":      d,
            "incident_type": itype.value,
            "severity":     severity.value,
            "description":  None,
            "days_lost":    days_lost,
        })

    return rows


def _gen_inspections(project: DimProject, start: date, end: date) -> list:
    """Monthly safety, quality, and code inspections throughout the project."""
    rows = []
    inspector_pool = [fake.name() for _ in range(8)]

    current_month_start = add_months(start, 1)
    while current_month_start <= end:
        # Target the 15th; nudge off weekends
        d = current_month_start.replace(
            day=min(15, calendar.monthrange(current_month_start.year,
                                            current_month_start.month)[1])
        )
        while d.weekday() >= 5:
            d += timedelta(days=1)
        if d > end:
            break

        for itype in InspectionType:
            result = random.choices(
                [InspectionResult.PASS, InspectionResult.CONDITIONAL, InspectionResult.FAIL],
                weights=[85, 8, 7],
            )[0]
            rows.append({
                "project_id":      project.id,
                "date_id":         d,
                "inspection_type": itype.value,
                "result":          result.value,
                "inspector_name":  random.choice(inspector_pool),
                "notes":           None,
            })

        current_month_start = add_months(current_month_start, 1)

    return rows


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------


def clear_fact_tables(session: Session) -> None:
    """Delete all fact rows while leaving dimension tables intact."""
    fact_models = [
        FactProjectCost, FactLaborHour, FactEquipmentUsage,
        FactMaterialDelivery, FactSafetyIncident, FactInspection,
    ]
    for model in fact_models:
        session.execute(delete(model))
    session.commit()
    print("  Cleared existing fact data.")


def main() -> None:
    print("=" * 65)
    print("  Construction Analytics Platform — Data Generator")
    print("=" * 65)

    engine = get_engine()
    Base.metadata.create_all(engine)   # idempotent: creates any missing tables

    with Session(engine) as session:

        # ── 1. DIMENSION TABLES ───────────────────────────────────────────
        print("\n[1/3] Dimension tables")
        generate_dim_dates(session)
        projects       = generate_projects(session)
        subcontractors = generate_subcontractors(session)
        equipment      = generate_equipment(session)
        employees      = generate_employees(session)
        materials      = generate_materials(session)

        # ── 2. CLEAR OLD FACT DATA ────────────────────────────────────────
        print("\n[2/3] Clearing existing fact data")
        clear_fact_tables(session)

        # ── 3. FACT TABLES ────────────────────────────────────────────────
        print(f"\n[3/3] Generating fact data for {len(projects)} projects")
        print("-" * 65)

        totals = dict(costs=0, labor=0, equip=0, deliveries=0, incidents=0, inspections=0)

        for i, project in enumerate(projects, 1):
            rng = _effective_range(project)
            label = f"[{i:02d}/{len(projects)}] {project.project_name[:38]:<38}"

            if rng is None:
                print(f"  {label}  PLANNING — skipped")
                continue

            start, end = rng
            print(f"  {label}  {start} → {end}", end="  ", flush=True)

            costs      = _gen_costs(project, start, end, subcontractors)
            labor      = _gen_labor(project, start, end, employees)
            equip      = _gen_equipment_usage(project, start, end, equipment, employees)
            deliveries = _gen_material_deliveries(project, start, end, materials)
            incidents  = _gen_safety_incidents(project, start, end)
            inspections = _gen_inspections(project, start, end)

            bulk_insert(session, FactProjectCost,       costs)
            bulk_insert(session, FactLaborHour,         labor)
            bulk_insert(session, FactEquipmentUsage,    equip)
            bulk_insert(session, FactMaterialDelivery,  deliveries)
            if incidents:
                bulk_insert(session, FactSafetyIncident, incidents)
            if inspections:
                bulk_insert(session, FactInspection,     inspections)

            totals["costs"]       += len(costs)
            totals["labor"]       += len(labor)
            totals["equip"]       += len(equip)
            totals["deliveries"]  += len(deliveries)
            totals["incidents"]   += len(incidents)
            totals["inspections"] += len(inspections)

            print(f"costs={len(costs):,}  labor={len(labor):,}  "
                  f"equip={len(equip):,}  mat={len(deliveries):,}")

        # ── SUMMARY ───────────────────────────────────────────────────────
        total_fact = sum(totals.values())
        date_rows  = (DATA_END - DATA_START).days + 1

        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        print(f"  {'dim_dates':<30} {date_rows:>10,}")
        print(f"  {'dim_projects':<30} {len(projects):>10,}")
        print(f"  {'dim_subcontractors':<30} {len(subcontractors):>10,}")
        print(f"  {'dim_equipment':<30} {len(equipment):>10,}")
        print(f"  {'dim_employees':<30} {len(employees):>10,}")
        print(f"  {'dim_materials':<30} {len(materials):>10,}")
        print(f"  {'─' * 42}")
        print(f"  {'fact_project_costs':<30} {totals['costs']:>10,}")
        print(f"  {'fact_labor_hours':<30} {totals['labor']:>10,}")
        print(f"  {'fact_equipment_usage':<30} {totals['equip']:>10,}")
        print(f"  {'fact_material_deliveries':<30} {totals['deliveries']:>10,}")
        print(f"  {'fact_safety_incidents':<30} {totals['incidents']:>10,}")
        print(f"  {'fact_inspections':<30} {totals['inspections']:>10,}")
        print(f"  {'─' * 42}")
        print(f"  {'Total fact rows':<30} {total_fact:>10,}")
        print("=" * 65)
        print(f"\n  Database: {engine.url}")
        print("  Done.\n")


if __name__ == "__main__":
    main()
