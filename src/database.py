"""
database.py — SQLite database setup for the Construction Analytics Platform.

Star schema with dimension and fact tables, built with SQLAlchemy 2.0+.
Run directly to initialise the database: python src/database.py
"""

from __future__ import annotations

import enum
from datetime import date
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------------------------------------------------------
# Engine & database path
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "construction.db"
_DATABASE_URL = f"sqlite:///{DB_PATH}"


def get_engine(database_url: str = _DATABASE_URL):
    """Return a SQLAlchemy engine connected to the SQLite database."""
    engine = create_engine(database_url, echo=False)

    # SQLite does not enforce foreign keys by default — enable per connection
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


# ---------------------------------------------------------------------------
# ORM base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProjectType(str, enum.Enum):
    RESIDENTIAL = "RESIDENTIAL"
    COMMERCIAL = "COMMERCIAL"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class ProjectStatus(str, enum.Enum):
    PLANNING = "PLANNING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    ON_HOLD = "ON_HOLD"


class Trade(str, enum.Enum):
    ELECTRICAL = "ELECTRICAL"
    PLUMBING = "PLUMBING"
    HVAC = "HVAC"
    FRAMING = "FRAMING"
    CONCRETE = "CONCRETE"
    ROOFING = "ROOFING"
    PAINTING = "PAINTING"
    LANDSCAPING = "LANDSCAPING"
    GENERAL = "GENERAL"


class EquipmentType(str, enum.Enum):
    EXCAVATOR = "EXCAVATOR"
    CRANE = "CRANE"
    BULLDOZER = "BULLDOZER"
    TRUCK = "TRUCK"
    FORKLIFT = "FORKLIFT"
    GENERATOR = "GENERATOR"
    COMPACTOR = "COMPACTOR"
    OTHER = "OTHER"


class EquipmentStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    MAINTENANCE = "MAINTENANCE"
    RETIRED = "RETIRED"


class CertificationLevel(str, enum.Enum):
    APPRENTICE = "APPRENTICE"
    JOURNEYMAN = "JOURNEYMAN"
    MASTER = "MASTER"


class EmployeeStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class MaterialCategory(str, enum.Enum):
    LUMBER = "LUMBER"
    CONCRETE = "CONCRETE"
    STEEL = "STEEL"
    ELECTRICAL = "ELECTRICAL"
    PLUMBING = "PLUMBING"
    INSULATION = "INSULATION"
    FINISHES = "FINISHES"
    OTHER = "OTHER"


class CostCategory(str, enum.Enum):
    LABOR = "LABOR"
    MATERIALS = "MATERIALS"
    EQUIPMENT = "EQUIPMENT"
    SUBCONTRACTOR = "SUBCONTRACTOR"
    OTHER = "OTHER"


class DeliveryStatus(str, enum.Enum):
    ON_TIME = "ON_TIME"
    DELAYED = "DELAYED"


class IncidentType(str, enum.Enum):
    INJURY = "INJURY"
    NEAR_MISS = "NEAR_MISS"
    VIOLATION = "VIOLATION"


class Severity(str, enum.Enum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


class InspectionType(str, enum.Enum):
    SAFETY = "SAFETY"
    QUALITY = "QUALITY"
    CODE = "CODE"


class InspectionResult(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    CONDITIONAL = "CONDITIONAL"


# ---------------------------------------------------------------------------
# DIMENSION TABLES
# ---------------------------------------------------------------------------


class DimProject(Base):
    __tablename__ = "dim_projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_name: Mapped[str] = mapped_column(String(200), nullable=False)
    project_type: Mapped[str] = mapped_column(Enum(ProjectType), nullable=False)
    contract_value: Mapped[float] = mapped_column(Float, nullable=False)
    planned_start_date: Mapped[Optional[date]] = mapped_column(Date)
    planned_end_date: Mapped[Optional[date]] = mapped_column(Date)
    actual_start_date: Mapped[Optional[date]] = mapped_column(Date)
    actual_end_date: Mapped[Optional[date]] = mapped_column(Date)
    status: Mapped[str] = mapped_column(
        Enum(ProjectStatus), nullable=False, default=ProjectStatus.PLANNING
    )
    client_name: Mapped[str] = mapped_column(String(200), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(300))
    square_footage: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships to fact tables
    project_costs: Mapped[list[FactProjectCost]] = relationship(back_populates="project")
    labor_hours: Mapped[list[FactLaborHour]] = relationship(back_populates="project")
    equipment_usage: Mapped[list[FactEquipmentUsage]] = relationship(back_populates="project")
    material_deliveries: Mapped[list[FactMaterialDelivery]] = relationship(back_populates="project")
    safety_incidents: Mapped[list[FactSafetyIncident]] = relationship(back_populates="project")
    inspections: Mapped[list[FactInspection]] = relationship(back_populates="project")

    __table_args__ = (
        Index("ix_dim_projects_status", "status"),
        Index("ix_dim_projects_project_type", "project_type"),
    )


class DimSubcontractor(Base):
    __tablename__ = "dim_subcontractors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_name: Mapped[str] = mapped_column(String(200), nullable=False)
    trade: Mapped[str] = mapped_column(Enum(Trade), nullable=False)
    contact_person: Mapped[Optional[str]] = mapped_column(String(150))
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    email: Mapped[Optional[str]] = mapped_column(String(150))
    rating: Mapped[Optional[float]] = mapped_column(Float)  # 1–5 scale
    years_in_business: Mapped[Optional[int]] = mapped_column(Integer)

    project_costs: Mapped[list[FactProjectCost]] = relationship(back_populates="subcontractor")

    __table_args__ = (Index("ix_dim_subcontractors_trade", "trade"),)


class DimEquipment(Base):
    __tablename__ = "dim_equipment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_name: Mapped[str] = mapped_column(String(200), nullable=False)
    equipment_type: Mapped[str] = mapped_column(Enum(EquipmentType), nullable=False)
    purchase_date: Mapped[Optional[date]] = mapped_column(Date)
    purchase_cost: Mapped[Optional[float]] = mapped_column(Float)
    hourly_rate: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(
        Enum(EquipmentStatus), nullable=False, default=EquipmentStatus.ACTIVE
    )

    equipment_usage: Mapped[list[FactEquipmentUsage]] = relationship(back_populates="equipment")

    __table_args__ = (
        Index("ix_dim_equipment_status", "status"),
        Index("ix_dim_equipment_type", "equipment_type"),
    )


class DimEmployee(Base):
    __tablename__ = "dim_employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_name: Mapped[str] = mapped_column(String(150), nullable=False)
    trade: Mapped[str] = mapped_column(Enum(Trade), nullable=False)
    hourly_rate: Mapped[float] = mapped_column(Float, nullable=False)
    hire_date: Mapped[Optional[date]] = mapped_column(Date)
    certification_level: Mapped[str] = mapped_column(Enum(CertificationLevel), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(EmployeeStatus), nullable=False, default=EmployeeStatus.ACTIVE
    )

    labor_hours: Mapped[list[FactLaborHour]] = relationship(back_populates="employee")
    # Operator FK in fact_equipment_usage points here too; disambiguate with foreign_keys
    equipment_operated: Mapped[list[FactEquipmentUsage]] = relationship(
        back_populates="operator",
        foreign_keys="FactEquipmentUsage.operator_id",
    )

    __table_args__ = (
        Index("ix_dim_employees_status", "status"),
        Index("ix_dim_employees_trade", "trade"),
    )


class DimMaterial(Base):
    __tablename__ = "dim_materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    material_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(Enum(MaterialCategory), nullable=False)
    unit: Mapped[str] = mapped_column(String(50), nullable=False)  # ton/yard/foot/piece/etc.
    unit_cost: Mapped[float] = mapped_column(Float, nullable=False)
    supplier: Mapped[Optional[str]] = mapped_column(String(200))

    material_deliveries: Mapped[list[FactMaterialDelivery]] = relationship(
        back_populates="material"
    )

    __table_args__ = (Index("ix_dim_materials_category", "category"),)


class DimDate(Base):
    """Pre-populated date dimension — use populate_dim_dates() to fill."""

    __tablename__ = "dim_dates"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    quarter: Mapped[int] = mapped_column(Integer, nullable=False)  # 1–4
    month: Mapped[int] = mapped_column(Integer, nullable=False)    # 1–12
    month_name: Mapped[str] = mapped_column(String(20), nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)     # ISO week number
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)  # 0=Mon … 6=Sun
    is_weekend: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_holiday: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    project_costs: Mapped[list[FactProjectCost]] = relationship(back_populates="date")
    labor_hours: Mapped[list[FactLaborHour]] = relationship(back_populates="date")
    equipment_usage: Mapped[list[FactEquipmentUsage]] = relationship(back_populates="date")
    material_deliveries: Mapped[list[FactMaterialDelivery]] = relationship(back_populates="date")
    safety_incidents: Mapped[list[FactSafetyIncident]] = relationship(back_populates="date")
    inspections: Mapped[list[FactInspection]] = relationship(back_populates="date")

    __table_args__ = (
        Index("ix_dim_dates_year_month", "year", "month"),
        Index("ix_dim_dates_year_quarter", "year", "quarter"),
    )


# ---------------------------------------------------------------------------
# FACT TABLES
# ---------------------------------------------------------------------------


class FactProjectCost(Base):
    __tablename__ = "fact_project_costs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("dim_projects.id"), nullable=False)
    date_id: Mapped[date] = mapped_column(ForeignKey("dim_dates.date"), nullable=False)
    cost_category: Mapped[str] = mapped_column(Enum(CostCategory), nullable=False)
    subcontractor_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("dim_subcontractors.id"), nullable=True
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    budgeted_amount: Mapped[Optional[float]] = mapped_column(Float)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    project: Mapped[DimProject] = relationship(back_populates="project_costs")
    date: Mapped[DimDate] = relationship(back_populates="project_costs")
    subcontractor: Mapped[Optional[DimSubcontractor]] = relationship(
        back_populates="project_costs"
    )

    __table_args__ = (
        Index("ix_fact_project_costs_project_id", "project_id"),
        Index("ix_fact_project_costs_date_id", "date_id"),
        Index("ix_fact_project_costs_cost_category", "cost_category"),
    )


class FactLaborHour(Base):
    __tablename__ = "fact_labor_hours"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("dim_projects.id"), nullable=False)
    employee_id: Mapped[int] = mapped_column(ForeignKey("dim_employees.id"), nullable=False)
    date_id: Mapped[date] = mapped_column(ForeignKey("dim_dates.date"), nullable=False)
    hours_worked: Mapped[float] = mapped_column(Float, nullable=False)
    hourly_rate: Mapped[float] = mapped_column(Float, nullable=False)  # snapshotted at time of entry
    task_description: Mapped[Optional[str]] = mapped_column(Text)

    project: Mapped[DimProject] = relationship(back_populates="labor_hours")
    employee: Mapped[DimEmployee] = relationship(back_populates="labor_hours")
    date: Mapped[DimDate] = relationship(back_populates="labor_hours")

    __table_args__ = (
        Index("ix_fact_labor_hours_project_id", "project_id"),
        Index("ix_fact_labor_hours_employee_id", "employee_id"),
        Index("ix_fact_labor_hours_date_id", "date_id"),
    )


class FactEquipmentUsage(Base):
    __tablename__ = "fact_equipment_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("dim_projects.id"), nullable=False)
    equipment_id: Mapped[int] = mapped_column(ForeignKey("dim_equipment.id"), nullable=False)
    date_id: Mapped[date] = mapped_column(ForeignKey("dim_dates.date"), nullable=False)
    hours_used: Mapped[float] = mapped_column(Float, nullable=False)
    fuel_cost: Mapped[Optional[float]] = mapped_column(Float)
    maintenance_cost: Mapped[Optional[float]] = mapped_column(Float)
    operator_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("dim_employees.id"), nullable=True
    )

    project: Mapped[DimProject] = relationship(back_populates="equipment_usage")
    equipment: Mapped[DimEquipment] = relationship(back_populates="equipment_usage")
    date: Mapped[DimDate] = relationship(back_populates="equipment_usage")
    operator: Mapped[Optional[DimEmployee]] = relationship(
        back_populates="equipment_operated",
        foreign_keys=[operator_id],
    )

    __table_args__ = (
        Index("ix_fact_equipment_usage_project_id", "project_id"),
        Index("ix_fact_equipment_usage_equipment_id", "equipment_id"),
        Index("ix_fact_equipment_usage_date_id", "date_id"),
    )


class FactMaterialDelivery(Base):
    __tablename__ = "fact_material_deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("dim_projects.id"), nullable=False)
    material_id: Mapped[int] = mapped_column(ForeignKey("dim_materials.id"), nullable=False)
    date_id: Mapped[date] = mapped_column(ForeignKey("dim_dates.date"), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False)
    delivery_status: Mapped[str] = mapped_column(Enum(DeliveryStatus), nullable=False)

    project: Mapped[DimProject] = relationship(back_populates="material_deliveries")
    material: Mapped[DimMaterial] = relationship(back_populates="material_deliveries")
    date: Mapped[DimDate] = relationship(back_populates="material_deliveries")

    __table_args__ = (
        Index("ix_fact_material_deliveries_project_id", "project_id"),
        Index("ix_fact_material_deliveries_date_id", "date_id"),
        Index("ix_fact_material_deliveries_delivery_status", "delivery_status"),
    )


class FactSafetyIncident(Base):
    __tablename__ = "fact_safety_incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("dim_projects.id"), nullable=False)
    date_id: Mapped[date] = mapped_column(ForeignKey("dim_dates.date"), nullable=False)
    incident_type: Mapped[str] = mapped_column(Enum(IncidentType), nullable=False)
    severity: Mapped[str] = mapped_column(Enum(Severity), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    days_lost: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    project: Mapped[DimProject] = relationship(back_populates="safety_incidents")
    date: Mapped[DimDate] = relationship(back_populates="safety_incidents")

    __table_args__ = (
        Index("ix_fact_safety_incidents_project_id", "project_id"),
        Index("ix_fact_safety_incidents_severity", "severity"),
        Index("ix_fact_safety_incidents_incident_type", "incident_type"),
    )


class FactInspection(Base):
    __tablename__ = "fact_inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("dim_projects.id"), nullable=False)
    date_id: Mapped[date] = mapped_column(ForeignKey("dim_dates.date"), nullable=False)
    inspection_type: Mapped[str] = mapped_column(Enum(InspectionType), nullable=False)
    result: Mapped[str] = mapped_column(Enum(InspectionResult), nullable=False)
    inspector_name: Mapped[Optional[str]] = mapped_column(String(150))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    project: Mapped[DimProject] = relationship(back_populates="inspections")
    date: Mapped[DimDate] = relationship(back_populates="inspections")

    __table_args__ = (
        Index("ix_fact_inspections_project_id", "project_id"),
        Index("ix_fact_inspections_result", "result"),
        Index("ix_fact_inspections_inspection_type", "inspection_type"),
    )


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


def init_db(engine=None) -> None:
    """Create all tables. Safe to call multiple times (uses CREATE IF NOT EXISTS)."""
    if engine is None:
        engine = get_engine()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    print(f"Database initialised at: {DB_PATH}")


if __name__ == "__main__":
    init_db()
