import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
from fastapi import FastAPI, Query, Path, Body
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean,
    create_engine, UniqueConstraint, Index, CheckConstraint, text, Date
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship
)
from sqlalchemy.sql import func
import datetime
import json

DATABASE_PATH = os.environ.get("DATABASE_PATH")
if not DATABASE_PATH:
    DATABASE_PATH = "sqlite:///xxxx.db"

Base = declarative_base()
engine = create_engine('sqlite:///./outputs/databases/gofundme.db', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String)
    profile_json = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)

class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)

class Campaign(Base):
    __tablename__ = "campaigns"
    id = Column(Integer, primary_key=True)
    owner_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    title = Column(String, nullable=False)
    beneficiary_name = Column(String, nullable=False)
    location_city = Column(String, nullable=False)
    location_region = Column(String, nullable=False)
    location_country = Column(String, default="US", nullable=False)
    goal_amount_cents = Column(Integer, nullable=False)
    currency_code = Column(String, default="USD", nullable=False)
    status = Column(String, default="active", nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    __table_args__ = (
        UniqueConstraint("owner_user_id", "title", name="uix_campaigns_owner_title"),
        Index("idx_campaigns_owner", "owner_user_id"),
        Index("idx_campaigns_category_location", "category_id", "location_city", "location_region", "status"),
        Index("idx_campaigns_status", "status"),
    )

class CampaignMedia(Base):
    __tablename__ = "campaign_media"
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    alt_text = Column(String)
    sort_order = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    __table_args__ = (
        UniqueConstraint("campaign_id", "filename", name="uix_campaign_media_filename"),
        Index("idx_campaign_media_campaign_sort", "campaign_id", "sort_order", "id"),
    )

class CampaignUpdate(Base):
    __tablename__ = "campaign_updates"
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    author_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    __table_args__ = (
        Index("idx_campaign_updates_campaign_created", "campaign_id", "created_at".replace('DESC', '')),
    )

class DonationIntent(Base):
    __tablename__ = "donation_intents"
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True)
    donor_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    donor_name = Column(String, nullable=False)
    message = Column(Text)
    is_anonymous = Column(Integer, nullable=False, default=0)
    kind = Column(String, nullable=False)
    currency_code = Column(String, nullable=False, default="USD")
    amount_cents = Column(Integer, nullable=False)
    start_date = Column(Date, nullable=True)
    interval_unit = Column(String, nullable=True)
    interval_count = Column(Integer, nullable=True)
    status = Column(String, nullable=False, default="active")
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    __table_args__ = (
        Index("idx_donation_intents_campaign", "campaign_id"),
        Index("idx_donation_intents_donor_created", "donor_user_id", "created_at"),
        Index("idx_donation_intents_kind_status", "kind", "status"),
    )

class Donation(Base):
    __tablename__ = "donations"
    id = Column(Integer, primary_key=True)
    donation_intent_id = Column(Integer, ForeignKey("donation_intents.id", ondelete="SET NULL"), nullable=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True)
    donor_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    donor_name = Column(String, nullable=False)
    message = Column(Text)
    is_anonymous = Column(Integer, nullable=False, default=0)
    currency_code = Column(String, nullable=False, default="USD")
    amount_cents = Column(Integer, nullable=False)
    donated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    __table_args__ = (
        Index("idx_donations_campaign_donated", "campaign_id", "donated_at"),
        Index("idx_donations_donor_donated", "donor_user_id", "donated_at"),
        Index("idx_donations_intent", "donation_intent_id"),
    )

class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(Integer, primary_key=True)
    donation_id = Column(Integer, ForeignKey("donations.id", ondelete="CASCADE"), unique=True, nullable=False)
    receipt_number = Column(String, nullable=False, unique=True, index=True)
    issued_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False, index=True)
    total_amount_cents = Column(Integer, nullable=False)
    currency_code = Column(String, nullable=False, default="USD")

class PayoutMethod(Base):
    __tablename__ = "payout_methods"
    id = Column(Integer, primary_key=True)
    owner_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String, nullable=False)
    bank_name = Column(String, nullable=False)
    account_last4 = Column(String, nullable=False)
    account_type = Column(String, nullable=False)
    is_default = Column(Integer, nullable=False, default=0, index=True)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)

class CampaignPayoutSettings(Base):
    __tablename__ = "campaign_payout_settings"
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), primary_key=True)
    default_payout_method_id = Column(Integer, ForeignKey("payout_methods.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)

class Payout(Base):
    __tablename__ = "payouts"
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True)
    payout_method_id = Column(Integer, ForeignKey("payout_methods.id"), nullable=False, index=True)
    amount_cents = Column(Integer, nullable=False)
    currency_code = Column(String, nullable=False, default="USD")
    status = Column(String, nullable=False, default="pending")
    risk_level = Column(String, nullable=False, default="none")
    risk_flags_json = Column(Text)
    verification_required = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    __table_args__ = (
        Index("idx_payouts_campaign_created", "campaign_id", "created_at"),
        Index("idx_payouts_status", "status"),
        Index("idx_payouts_method", "payout_method_id"),
    )

class PayoutVerificationRequirement(Base):
    __tablename__ = "payout_verification_requirements"
    id = Column(Integer, primary_key=True)
    payout_id = Column(Integer, ForeignKey("payouts.id", ondelete="CASCADE"), nullable=False, index=True)
    requirement_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="required")
    notes = Column(Text)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    updated_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
    __table_args__ = (
        Index("idx_payout_verification_payout", "payout_id"),
        Index("idx_payout_verification_status", "status"),
    )

Base.metadata.create_all(engine)

app = FastAPI(title="GoFundMe simplified", version="1.0.0")

# Pydantic Models

class ImageMetadataItem(BaseModel):
    filename: str = Field(..., description="Image filename", example="maya-hospital.jpg")
    alt: str = Field(..., description="Alt text for image", example="Maya smiling in a hospital bed")

class CreateCampaignRequest(BaseModel):
    title: str = Field(..., description="Unique campaign title for the current user", example="Help Maya’s Recovery")
    category_name: str = Field(..., description="Category name for the campaign", example="Medical")
    goal_amount_usd: float = Field(..., description="Goal amount in USD", example=12000.0)
    beneficiary_name: str = Field(..., description="Beneficiary's name", example="Maya Hernandez")
    location_city: str = Field(..., description="Campaign city", example="Austin")
    location_region: str = Field(..., description="Campaign region/state", example="TX")
    image_metadata: List[ImageMetadataItem] = Field(..., description="Array of image filename and alt text objects", example=[{"filename":"maya-hospital.jpg","alt":"Maya smiling in a hospital bed"},{"filename":"family.jpg","alt":"Maya with her family at home"}])

class CampaignResponse(BaseModel):
    campaign_id: int = Field(..., description="Newly created campaign identifier", example=42)
    title: str = Field(..., description="Campaign title", example="Help Maya’s Recovery")
    status: str = Field(..., description="Campaign status", example="active")
    created_at: str = Field(..., description="Datetime of creation", example="2026-01-18T12:34:56Z")
    model_config = ConfigDict(from_attributes=True)

class UpdateCampaignGoalAndLocationRequest(BaseModel):
    goal_amount_usd: Optional[float] = Field(None, description="New goal amount in USD", example=15000.0)
    location_city: Optional[str] = Field(None, description="Updated city location", example="Round Rock")
    location_region: Optional[str] = Field(None, description="Updated region/state location", example="TX")

class UpdateCampaignGoalAndLocationResponse(BaseModel):
    campaign_id: int = Field(..., description="Campaign identifier", example=42)
    title: str = Field(..., description="Campaign title", example="Help Maya’s Recovery")
    goal_amount_usd: float = Field(..., description="Updated goal in USD", example=15000.0)
    location_city: str = Field(..., description="Updated city", example="Round Rock")
    location_region: str = Field(..., description="Updated region/state", example="TX")
    updated_at: str = Field(..., description="Datetime campaign updated", example="2026-02-02T10:15:18Z")
    model_config = ConfigDict(from_attributes=True)

class CampaignUpdateRequest(BaseModel):
    title: str = Field(..., description="Update title", example="Surgery scheduled")
    body: str = Field(..., description="Update message/body text", example="Maya’s surgery is scheduled for March 5th; thank you for the support—your donations are covering pre-op tests and travel.")

class CampaignUpdateResponse(BaseModel):
    update_id: int = Field(..., description="Identifier for the new update", example=15)
    campaign_id: int = Field(..., description="Related campaign ID", example=42)
    title: str = Field(..., description="Update title", example="Surgery scheduled")
    body: str = Field(..., description="Update body", example="Maya’s surgery is scheduled for March 5th; thank you for the support—your donations are covering pre-op tests and travel.")
    created_at: str = Field(..., description="Datetime update posted", example="2026-02-03T05:15:10Z")
    model_config = ConfigDict(from_attributes=True)

class SearchCampaignsByCategoryLocationResponseItem(BaseModel):
    campaign_id: int = Field(..., description="Campaign identifier", example=13)
    title: str = Field(..., description="Campaign title", example="Rescue Dog Treatment")
    category_name: str = Field(..., description="Category name", example="Animals")
    location_city: str = Field(..., description="City", example="Seattle")
    location_region: str = Field(..., description="Region/state", example="WA")
    goal_amount_usd: float = Field(..., description="Funding goal in USD", example=5000.0)
    amount_raised_usd: float = Field(..., description="Total donations raised in USD", example=4200.0)
    status: str = Field(..., description="Campaign status", example="active")

class SearchCampaignsByCategoryLocationResponse(BaseModel):
    campaigns: List[SearchCampaignsByCategoryLocationResponseItem] = Field(..., description="Campaigns found")

class CreateDonationIntentRequest(BaseModel):
    campaign_title: str = Field(..., description="Target campaign by title", example="Community Food Pantry Restock")
    donor_name: str = Field(..., description="Donor display name", example="Alex Kim")
    message: Optional[str] = Field(None, description="Public message from donor", example="Happy to help—thank you for feeding our neighbors!")
    amount_usd: float = Field(..., description="Amount to donate in USD", example=75.0)
    is_anonymous: bool = Field(..., description="If true, donor is anonymous", example=False)

class CreateDonationIntentResponse(BaseModel):
    donation_intent_id: int = Field(..., description="New donation intent ID", example=54)
    campaign_id: int = Field(..., description="Campaign identifier", example=12)
    donor_name: str = Field(..., description="Donor public name", example="Alex Kim")
    amount_usd: float = Field(..., description="Donation amount in USD", example=75.0)
    kind: str = Field(..., description="One-time or recurring", example="one_time")
    status: str = Field(..., description="Intent status", example="active")

class CreateRecurringDonationIntentRequest(BaseModel):
    campaign_title: str = Field(..., description="Target campaign by title", example="Wildfire Relief for Sonoma")
    donor_name: str = Field(..., description="Donor display name", example="Jordan Patel")
    message: Optional[str] = Field(None, description="Public message from donor", example="Stay strong—sending continued support.")
    amount_usd: float = Field(..., description="Monthly pledge amount in USD", example=20.0)
    is_anonymous: bool = Field(..., description="If true, donor is anonymous", example=True)
    start_date: str = Field(..., description="Recurring donation start date (YYYY-MM-DD)", example="2026-03-01")
    interval_unit: str = Field(..., description="Recurring interval unit", example="month")
    interval_count: int = Field(..., description="How many intervals between donations", example=1)

class CreateRecurringDonationIntentResponse(BaseModel):
    donation_intent_id: int = Field(..., description="New recurring donation intent ID", example=55)
    campaign_id: int = Field(..., description="Campaign identifier", example=14)
    donor_name: str = Field(..., description="Donor public name", example="Jordan Patel")
    amount_usd: float = Field(..., description="Donation amount in USD", example=20.0)
    kind: str = Field(..., description="Donation kind (recurring)", example="recurring")
    status: str = Field(..., description="Intent status", example="active")
    start_date: str = Field(..., description="Start date of recurrence", example="2026-03-01")
    interval_unit: str = Field(..., description="Interval unit", example="month")
    interval_count: int = Field(..., description="Interval count", example=1)

class FetchMyDonationHistoryDonation(BaseModel):
    donation_id: int = Field(..., description="Unique donation ID", example=111)
    campaign_title: str = Field(..., description="Title of the campaign donated to", example="Community Food Pantry Restock")
    amount_usd: float = Field(..., description="Donation amount in USD", example=75.0)
    donated_at: str = Field(..., description="Datetime donation was made", example="2026-01-20T16:26:00Z")
    kind: str = Field(..., description="Donation kind (one_time or recurring)", example="one_time")

class FetchMyDonationHistoryResponse(BaseModel):
    donations: List[FetchMyDonationHistoryDonation] = Field(..., description="Donations by the authenticated user")

class FetchDonationReceiptMetadataResponse(BaseModel):
    receipt_id: int = Field(..., description="Unique receipt identifier", example=75)
    receipt_number: str = Field(..., description="Receipt number", example="RCPT-20260120-111")
    issued_at: str = Field(..., description="Receipt issue datetime", example="2026-01-20T18:00:00Z")
    total_usd: float = Field(..., description="Total receipt amount in USD", example=75.0)

class CreateBankAccountPayoutMethodRequest(BaseModel):
    bank_name: str = Field(..., description="Name of the bank", example="Chase")
    account_last4: str = Field(..., description="Last 4 digits of account number", example="1234")
    account_type: str = Field(..., description="Account type: checking/savings", example="checking")
    is_default: Optional[bool] = Field(False, description="Set payout method as default", example=True)

class CreateBankAccountPayoutMethodResponse(BaseModel):
    payout_method_id: int = Field(..., description="New payout method identifier", example=8)
    bank_name: str = Field(..., description="Bank name", example="Chase")
    account_last4: str = Field(..., description="Last 4 of account number", example="1234")
    account_type: str = Field(..., description="Account type", example="checking")
    is_default: bool = Field(..., description="Is this payout method default for user", example=True)
    model_config = ConfigDict(from_attributes=True)

class SetCampaignDefaultPayoutMethodRequest(BaseModel):
    payout_method_id: int = Field(..., description="Default payout method identifier", example=8)

class SetCampaignDefaultPayoutMethodResponse(BaseModel):
    campaign_id: int = Field(..., description="Campaign identifier", example=42)
    default_payout_method_id: int = Field(..., description="Default payout method ID", example=8)
    updated_at: str = Field(..., description="Datetime payout settings updated", example="2026-01-19T11:15:23Z")

class RiskFlagItem(BaseModel):
    flag: str = Field(..., description="Risk, fraud, or AML flag", example="large_withdrawal")

class VerificationRequirementItem(BaseModel):
    requirement_type: str = Field(..., description="Verification type needed", example="identity_document")
    status: str = Field(..., description="Requirement status", example="required")
    notes: str = Field(..., description="Additional notes", example="Upload government ID")

class CreateCampaignPayoutToDefaultRequest(BaseModel):
    campaign_id: int = Field(..., description="Campaign identifier", example=42)
    amount_usd: float = Field(..., description="Amount to payout in USD", example=2500.0)

class CreateCampaignPayoutToDefaultResponse(BaseModel):
    payout_id: int = Field(..., description="New payout ID", example=25)
    campaign_id: int = Field(..., description="Campaign identifier", example=42)
    payout_method_id: int = Field(..., description="Payout method identifier", example=8)
    amount_usd: float = Field(..., description="Amount requested in USD", example=2500.0)
    status: str = Field(..., description="Payout status", example="pending")
    risk_level: str = Field(..., description="Risk level classification", example="low")
    risk_flags: List[RiskFlagItem] = Field(..., description="Risk/fraud flags")
    verification_required: bool = Field(..., description="Are verification docs required?", example=True)
    verification_requirements: List[VerificationRequirementItem] = Field(..., description="Verification requirements")

# Endpoint Implementations

@app.post(
    "/api/campaigns",
    summary="Create a new fundraising campaign",
    description="Create a campaign with a title, category, goal, beneficiary, location, and images for the current user.",
    tags=["campaigns"],
    operation_id="create_campaign",
    response_model=CampaignResponse,
)
async def create_campaign(
    req: CreateCampaignRequest,
):
    session = SessionLocal()
    user_id = 1
    category = session.query(Category).filter(Category.name == req.category_name).first()
    if not category:
        category = Category(name=req.category_name)
        session.add(category)
        session.commit()
        session.refresh(category)
    campaign = Campaign(
        owner_user_id=user_id,
        category_id=category.id,
        title=req.title,
        beneficiary_name=req.beneficiary_name,
        location_city=req.location_city,
        location_region=req.location_region,
        goal_amount_cents=int(round(req.goal_amount_usd * 100)),
    )
    session.add(campaign)
    session.commit()
    session.refresh(campaign)
    for idx, item in enumerate(req.image_metadata):
        media = CampaignMedia(
            campaign_id=campaign.id,
            filename=item.filename,
            alt_text=item.alt,
            sort_order=idx,
        )
        session.add(media)
    session.commit()
    resp = CampaignResponse(
        campaign_id=campaign.id,
        title=campaign.title,
        status=campaign.status,
        created_at=campaign.created_at.replace(microsecond=0).isoformat() + "Z",
    )
    session.close()
    return resp

@app.patch(
    "/api/campaigns/{campaign_id}",
    summary="Update goal amount and location of a campaign",
    description="Update a campaign's goal amount and location fields. Use for changing funding goals or event locations.",
    tags=["campaigns"],
    operation_id="update_campaign_goal_and_location",
    response_model=UpdateCampaignGoalAndLocationResponse,
)
async def update_campaign_goal_and_location(
    campaign_id: int = Path(..., description="Campaign identifier", example=42),
    req: UpdateCampaignGoalAndLocationRequest = Body(...),
):
    session = SessionLocal()
    user_id = 1
    campaign = session.query(Campaign).filter(Campaign.id == campaign_id, Campaign.owner_user_id == user_id).first()
    if req.goal_amount_usd is not None:
        campaign.goal_amount_cents = int(round(req.goal_amount_usd * 100))
    if req.location_city is not None:
        campaign.location_city = req.location_city
    if req.location_region is not None:
        campaign.location_region = req.location_region
    campaign.updated_at = datetime.datetime.utcnow()
    session.commit()
    resp = UpdateCampaignGoalAndLocationResponse(
        campaign_id=campaign.id,
        title=campaign.title,
        goal_amount_usd=campaign.goal_amount_cents / 100.0,
        location_city=campaign.location_city,
        location_region=campaign.location_region,
        updated_at=campaign.updated_at.replace(microsecond=0).isoformat() + "Z",
    )
    session.close()
    return resp

@app.post(
    "/api/campaigns/{campaign_id}/updates",
    summary="Post a campaign update",
    description="Add an update with a title and body to a campaign. Use for progress announcements or events.",
    tags=["campaigns", "campaign_updates"],
    operation_id="create_campaign_update",
    response_model=CampaignUpdateResponse,
)
async def create_campaign_update(
    campaign_id: int = Path(..., description="Campaign identifier", example=42),
    req: CampaignUpdateRequest = Body(...),
):
    session = SessionLocal()
    user_id = 1
    campaign = session.query(Campaign).filter(Campaign.id == campaign_id, Campaign.owner_user_id == user_id).first()
    update = CampaignUpdate(
        campaign_id=campaign_id,
        author_user_id=user_id,
        title=req.title,
        body=req.body,
    )
    session.add(update)
    session.commit()
    session.refresh(update)
    resp = CampaignUpdateResponse(
        update_id=update.id,
        campaign_id=update.campaign_id,
        title=update.title,
        body=update.body,
        created_at=update.created_at.replace(microsecond=0).isoformat() + "Z",
    )
    session.close()
    return resp

@app.get(
    "/api/campaigns/search",
    summary="Search campaigns by category and location",
    description="Find campaigns filtered by category name and city/region, sorted by amount raised descending.",
    tags=["campaigns"],
    operation_id="search_campaigns_by_category_location",
    response_model=SearchCampaignsByCategoryLocationResponse,
)
async def search_campaigns_by_category_location(
    category_name: str = Query(..., description="Campaign category name", example="Animals"),
    location_city: str = Query(..., description="Campaign city", example="Seattle"),
    location_region: str = Query(..., description="Campaign region/state", example="WA"),
    top_n: Optional[int] = Query(None, description="Limit to top N campaigns", example=5),
):
    session = SessionLocal()
    category = session.query(Category).filter(Category.name == category_name).first()
    campaigns = []
    if category:
        query = session.query(Campaign).filter(
            Campaign.category_id == category.id,
            Campaign.location_city == location_city,
            Campaign.location_region == location_region,
            Campaign.status == "active"
        )
        campaign_list = list(query)
        result = []
        for c in campaign_list:
            donation_sum = session.query(func.coalesce(func.sum(Donation.amount_cents), 0)).filter(Donation.campaign_id == c.id).scalar()
            result.append({
                "campaign": c,
                "amount_raised": donation_sum or 0,
            })
        result.sort(key=lambda x: x["amount_raised"], reverse=True)
        if top_n is not None:
            result = result[:top_n]
        for entry in result:
            c = entry["campaign"]
            amount_raised = entry["amount_raised"]
            campaigns.append(SearchCampaignsByCategoryLocationResponseItem(
                campaign_id=c.id,
                title=c.title,
                category_name=category_name,
                location_city=c.location_city,
                location_region=c.location_region,
                goal_amount_usd=c.goal_amount_cents / 100.0,
                amount_raised_usd=(amount_raised / 100.0) if amount_raised else 0.0,
                status=c.status,
            ))
    resp = SearchCampaignsByCategoryLocationResponse(campaigns=campaigns)
    session.close()
    return resp

@app.post(
    "/api/donation_intents",
    summary="Create a one-time donation intent for a campaign",
    description="Create a donation intent specifying campaign, donor name, amount, public message, and anonymity.",
    tags=["donations", "donation_intents"],
    operation_id="create_one_time_donation_intent",
    response_model=CreateDonationIntentResponse,
)
async def create_one_time_donation_intent(
    req: CreateDonationIntentRequest,
):
    session = SessionLocal()
    user_id = 1
    campaign = session.query(Campaign).filter(Campaign.title == req.campaign_title).first()
    donation_intent = DonationIntent(
        campaign_id=campaign.id,
        donor_user_id=user_id,
        donor_name=req.donor_name,
        message=req.message,
        is_anonymous=1 if req.is_anonymous else 0,
        kind="one_time",
        currency_code="USD",
        amount_cents=int(round(req.amount_usd * 100)),
        status="active",
    )
    session.add(donation_intent)
    session.commit()
    session.refresh(donation_intent)
    resp = CreateDonationIntentResponse(
        donation_intent_id=donation_intent.id,
        campaign_id=campaign.id,
        donor_name=donation_intent.donor_name,
        amount_usd=donation_intent.amount_cents / 100.0,
        kind=donation_intent.kind,
        status=donation_intent.status,
    )
    session.close()
    return resp

@app.post(
    "/api/donation_intents",
    summary="Create a recurring monthly donation intent for a campaign",
    description="Create a recurring donation intent including start date, interval info, donor details, and anonymity.",
    tags=["donations", "donation_intents"],
    operation_id="create_recurring_donation_intent",
    response_model=CreateRecurringDonationIntentResponse,
)
async def create_recurring_donation_intent(
    req: CreateRecurringDonationIntentRequest,
):
    session = SessionLocal()
    user_id = 1
    campaign = session.query(Campaign).filter(Campaign.title == req.campaign_title).first()
    start_date_obj = datetime.datetime.strptime(req.start_date, "%Y-%m-%d").date()
    donation_intent = DonationIntent(
        campaign_id=campaign.id,
        donor_user_id=user_id,
        donor_name=req.donor_name,
        message=req.message,
        is_anonymous=1 if req.is_anonymous else 0,
        kind="recurring",
        currency_code="USD",
        amount_cents=int(round(req.amount_usd * 100)),
        start_date=start_date_obj,
        interval_unit=req.interval_unit,
        interval_count=req.interval_count,
        status="active",
    )
    session.add(donation_intent)
    session.commit()
    session.refresh(donation_intent)
    resp = CreateRecurringDonationIntentResponse(
        donation_intent_id=donation_intent.id,
        campaign_id=campaign.id,
        donor_name=donation_intent.donor_name,
        amount_usd=donation_intent.amount_cents / 100.0,
        kind=donation_intent.kind,
        status=donation_intent.status,
        start_date=donation_intent.start_date.isoformat(),
        interval_unit=donation_intent.interval_unit,
        interval_count=donation_intent.interval_count,
    )
    session.close()
    return resp

@app.get(
    "/api/donations/history",
    summary="Fetch authenticated user's donation history for last 90 days",
    description="Get all donations made by the current user in the last 90 days, including campaign title, amount, date, and donation type.",
    tags=["donations"],
    operation_id="fetch_my_donation_history",
    response_model=FetchMyDonationHistoryResponse,
)
async def fetch_my_donation_history():
    session = SessionLocal()
    user_id = 1
    since = datetime.datetime.utcnow() - datetime.timedelta(days=90)
    query = session.query(Donation).filter(
        Donation.donor_user_id == user_id,
        Donation.donated_at >= since
    )
    items = []
    for donation in query.order_by(Donation.donated_at.desc()).all():
        campaign = session.query(Campaign).filter(Campaign.id == donation.campaign_id).first()
        d_intent = session.query(DonationIntent).filter(DonationIntent.id == donation.donation_intent_id).first() if donation.donation_intent_id else None
        kind = d_intent.kind if d_intent else "one_time"
        items.append(FetchMyDonationHistoryDonation(
            donation_id=donation.id,
            campaign_title=campaign.title,
            amount_usd=donation.amount_cents / 100.0,
            donated_at=donation.donated_at.replace(microsecond=0).isoformat() + "Z",
            kind=kind,
        ))
    resp = FetchMyDonationHistoryResponse(donations=items)
    session.close()
    return resp

@app.get(
    "/api/receipts",
    summary="Generate donation receipt metadata",
    description="Fetch receipt metadata for a specific donation to a campaign by date and amount.",
    tags=["receipts"],
    operation_id="fetch_donation_receipt_metadata",
    response_model=FetchDonationReceiptMetadataResponse,
)
async def fetch_donation_receipt_metadata(
    campaign_title: str = Query(..., description="Title of campaign receiving the donation", example="Community Food Pantry Restock"),
    donation_date: str = Query(..., description="Donation date, format YYYY-MM-DD", example="2026-01-20"),
    amount_usd: float = Query(..., description="Donation amount in USD", example=75.0),
):
    session = SessionLocal()
    user_id = 1
    campaign = session.query(Campaign).filter(Campaign.title == campaign_title).first()
    date_lower = datetime.datetime.strptime(donation_date, "%Y-%m-%d")
    date_upper = date_lower + datetime.timedelta(days=1)
    donation = session.query(Donation).filter(
        Donation.campaign_id == campaign.id,
        Donation.donor_user_id == user_id,
        Donation.amount_cents == int(round(amount_usd * 100)),
        Donation.donated_at >= date_lower,
        Donation.donated_at < date_upper,
    ).order_by(Donation.donated_at.desc()).first()
    receipt = session.query(Receipt).filter(Receipt.donation_id == donation.id).first()
    resp = FetchDonationReceiptMetadataResponse(
        receipt_id=receipt.id,
        receipt_number=receipt.receipt_number,
        issued_at=receipt.issued_at.replace(microsecond=0).isoformat() + "Z",
        total_usd=receipt.total_amount_cents / 100.0,
    )
    session.close()
    return resp

@app.post(
    "/api/payout_methods",
    summary="Add a new bank account payout method for user",
    description="Add a bank account payout method for the current user and return the details including default status.",
    tags=["payout_methods"],
    operation_id="create_bank_account_payout_method",
    response_model=CreateBankAccountPayoutMethodResponse,
)
async def create_bank_account_payout_method(
    req: CreateBankAccountPayoutMethodRequest,
):
    session = SessionLocal()
    user_id = 1
    is_default = bool(req.is_default)
    if is_default:
        session.query(PayoutMethod).filter(PayoutMethod.owner_user_id == user_id, PayoutMethod.is_default == 1).update({PayoutMethod.is_default:0})
        session.commit()
    payout_method = PayoutMethod(
        owner_user_id=user_id,
        type="bank_account",
        bank_name=req.bank_name,
        account_last4=req.account_last4,
        account_type=req.account_type,
        is_default=1 if is_default else 0,
    )
    session.add(payout_method)
    session.commit()
    session.refresh(payout_method)
    resp = CreateBankAccountPayoutMethodResponse(
        payout_method_id=payout_method.id,
        bank_name=payout_method.bank_name,
        account_last4=payout_method.account_last4,
        account_type=payout_method.account_type,
        is_default=bool(payout_method.is_default),
    )
    session.close()
    return resp

@app.put(
    "/api/campaigns/{campaign_id}/payout_settings",
    summary="Set default payout method for campaign",
    description="Configure payout settings for a campaign by setting the default payout method for withdrawals.",
    tags=["payout_methods", "campaigns"],
    operation_id="set_campaign_default_payout_method",
    response_model=SetCampaignDefaultPayoutMethodResponse,
)
async def set_campaign_default_payout_method(
    campaign_id: int = Path(..., description="Campaign identifier", example=42),
    req: SetCampaignDefaultPayoutMethodRequest = Body(...),
):
    session = SessionLocal()
    user_id = 1
    campaign = session.query(Campaign).filter(Campaign.id == campaign_id, Campaign.owner_user_id == user_id).first()
    s = session.query(CampaignPayoutSettings).filter(CampaignPayoutSettings.campaign_id == campaign_id).first()
    t = datetime.datetime.utcnow()
    if s:
        s.default_payout_method_id = req.payout_method_id
        s.updated_at = t
    else:
        s = CampaignPayoutSettings(
            campaign_id=campaign_id,
            default_payout_method_id=req.payout_method_id,
            created_at=t,
            updated_at=t,
        )
        session.add(s)
    session.commit()
    resp = SetCampaignDefaultPayoutMethodResponse(
        campaign_id=campaign_id,
        default_payout_method_id=req.payout_method_id,
        updated_at=s.updated_at.replace(microsecond=0).isoformat() + "Z",
    )
    session.close()
    return resp

@app.post(
    "/api/payouts",
    summary="Request a payout from a campaign to default bank account",
    description="Request a payout for a campaign to the default payout method. Returns payout status and any risk/verification requirements.",
    tags=["payouts", "campaigns"],
    operation_id="create_campaign_payout_to_default",
    response_model=CreateCampaignPayoutToDefaultResponse,
)
async def create_campaign_payout_to_default(
    req: CreateCampaignPayoutToDefaultRequest,
):
    session = SessionLocal()
    user_id = 1
    campaign = session.query(Campaign).filter(Campaign.id == req.campaign_id, Campaign.owner_user_id == user_id).first()
    ps = session.query(CampaignPayoutSettings).filter(CampaignPayoutSettings.campaign_id == req.campaign_id).first()
    payout_method_id = ps.default_payout_method_id
    amount_cents = int(round(req.amount_usd * 100))
    # Simulate risk
    risk_level = "low"
    risk_flags_list = []
    verification_required = False
    verification_requirements = []
    if req.amount_usd >= 2000.0:
        risk_level = "medium"
        risk_flags_list.append("large_withdrawal")
        verification_required = True
        v = PayoutVerificationRequirement(
            requirement_type="identity_document",
            status="required",
            notes="Upload government ID",
            payout_id=None,
        )
        verification_requirements.append(v)
    payout = Payout(
        campaign_id=req.campaign_id,
        payout_method_id=payout_method_id,
        amount_cents=amount_cents,
        status="pending",
        currency_code="USD",
        risk_level=risk_level,
        risk_flags_json=json.dumps(risk_flags_list),
        verification_required=1 if verification_required else 0,
    )
    session.add(payout)
    session.commit()
    session.refresh(payout)
    vr_items = []
    if verification_required:
        v.payout_id = payout.id
        session.add(v)
        session.commit()
        session.refresh(v)
        vr_items.append(VerificationRequirementItem(requirement_type=v.requirement_type, status=v.status, notes=v.notes))
    resp = CreateCampaignPayoutToDefaultResponse(
        payout_id=payout.id,
        campaign_id=req.campaign_id,
        payout_method_id=payout_method_id,
        amount_usd=amount_cents / 100.0,
        status=payout.status,
        risk_level=risk_level,
        risk_flags=[RiskFlagItem(flag=f) for f in risk_flags_list],
        verification_required=bool(payout.verification_required),
        verification_requirements=vr_items,
    )
    session.close()
    return resp

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    import os
    host = os.environ.get('HOST', '127.0.0.1')
    port = os.environ.get('PORT', 8001)
    print(f'Server starting on port={port}')
    from fastapi_mcp import FastApiMCP
    mcp = FastApiMCP(app)
    mcp.mount_http()
    print("MCP server enabled, please visit http://127.0.0.1:8001/mcp for the MCP service")
    uvicorn.run(app, host=host, port=int(port))
