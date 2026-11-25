"""One-off: generate sample PDFs for Module 4b eval fixtures.

Run with: uv run --with reportlab python scripts/generate_fixture_pdfs.py
Outputs to: evals/fixtures/pdfs/
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "evals" / "fixtures" / "pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=20, spaceAfter=12
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=14, spaceAfter=8
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=11, leading=15
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontSize=9, textColor=colors.grey
        ),
        "muted": ParagraphStyle(
            "muted", parent=base["BodyText"], fontSize=10, textColor=colors.grey
        ),
    }


def build_concert_ticket() -> None:
    path = OUT_DIR / "concert_ticket.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    )
    s = _styles()
    story = [
        Paragraph("MELODY HALL", s["h1"]),
        Paragraph("Your e-ticket — please show at entry", s["muted"]),
        Spacer(1, 18),
        Paragraph("<b>The Night Hours — Live in Concert</b>", s["h2"]),
        Spacer(1, 6),
        Table(
            [
                ["Date", "Saturday, 14 March 2027"],
                ["Doors open", "7:00 PM"],
                ["Show starts", "8:00 PM"],
                ["Venue", "Melody Hall, 42 River Street, Bristol BS1 4QA"],
                ["Seat", "Stalls — Row G, Seat 18"],
                ["Booking ref", "MH-2027-884201"],
            ],
            colWidths=[1.6 * inch, 4.0 * inch],
            style=TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 11),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]),
        ),
        Spacer(1, 18),
        Paragraph(
            "Bag policy: small bags only (A4 size or smaller). No photography "
            "during the performance. Latecomers will be seated at a suitable "
            "break in the programme.",
            s["body"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Order placed on 02 Nov 2026 · Ticket holder: A. Patel · "
            "Customer services: box.office@melodyhall.example",
            s["small"],
        ),
    ]
    doc.build(story)
    print(f"wrote {path}")


def build_dentist_reminder() -> None:
    path = OUT_DIR / "dentist_reminder.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    )
    s = _styles()
    story = [
        Paragraph("Riverside Dental Practice", s["h1"]),
        Paragraph("17 Willow Lane · Cambridge CB2 1AA · 01223 555 0144", s["muted"]),
        Spacer(1, 20),
        Paragraph("<b>Appointment reminder</b>", s["h2"]),
        Spacer(1, 6),
        Paragraph(
            "Dear Ms Patel,<br/><br/>"
            "This is a reminder of your upcoming appointment with "
            "<b>Dr Helen Carter</b>.",
            s["body"],
        ),
        Spacer(1, 12),
        Table(
            [
                ["When", "Tuesday 9 February 2027, 10:30 AM"],
                ["Duration", "30 minutes"],
                ["Appointment type", "Routine check-up and clean"],
                ["Location", "Riverside Dental Practice, Room 2"],
            ],
            colWidths=[1.6 * inch, 4.2 * inch],
            style=TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 11),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ]),
        ),
        Spacer(1, 16),
        Paragraph(
            "Please arrive 5 minutes early to complete a short medical update "
            "form. If you need to reschedule, please give at least 24 hours' "
            "notice to avoid a cancellation fee.",
            s["body"],
        ),
        Spacer(1, 20),
        Paragraph("Kind regards,<br/>Riverside Dental Team", s["body"]),
    ]
    doc.build(story)
    print(f"wrote {path}")


def build_conference_agenda() -> None:
    path = OUT_DIR / "conference_agenda.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )
    s = _styles()

    def session_table(rows: list[list[str]]) -> Table:
        return Table(
            rows,
            colWidths=[1.1 * inch, 1.6 * inch, 3.6 * inch],
            style=TableStyle([
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]),
        )

    story = [
        Paragraph("NorthCoast PM Summit 2027", s["h1"]),
        Paragraph("18 – 19 May 2027 · The Harbour Hotel, Edinburgh EH6 6QW",
                  s["muted"]),
        Spacer(1, 16),
        Paragraph(
            "Welcome to the fifth annual NorthCoast PM Summit. Two days of "
            "workshops, keynotes and roundtables for product leaders working "
            "on AI-enabled software. This programme is subject to minor "
            "changes — please check in at the registration desk on arrival.",
            s["body"],
        ),
        Spacer(1, 18),
        Paragraph("<b>Day 1 — Monday 18 May 2027</b>", s["h2"]),
        session_table([
            ["Time", "Room", "Session"],
            ["09:00 – 09:30", "Foyer", "Registration and coffee"],
            ["09:30 – 10:30", "Main Hall", "Opening keynote: Shipping agents that users actually trust — Priya Rao"],
            ["10:45 – 12:00", "Hall A", "Workshop: Writing evals that catch real regressions"],
            ["10:45 – 12:00", "Hall B", "Panel: Pricing and packaging AI features"],
            ["12:00 – 13:00", "Foyer", "Lunch"],
            ["13:00 – 14:30", "Hall A", "Workshop: From prototype to prod — infra for small teams"],
            ["14:45 – 15:45", "Main Hall", "Fireside chat: Scaling PM teams in the AI era"],
            ["18:30 – late",  "Rooftop", "Welcome drinks and dinner"],
        ]),
        PageBreak(),
        Paragraph("<b>Day 2 — Tuesday 19 May 2027</b>", s["h2"]),
        session_table([
            ["Time", "Room", "Session"],
            ["09:00 – 10:00", "Main Hall", "Keynote: What great PM looks like when the model is the product"],
            ["10:15 – 11:30", "Hall A", "Workshop: User research for agentic products"],
            ["10:15 – 11:30", "Hall B", "Case study: Replacing a tool with an agent — what we learned"],
            ["11:45 – 12:45", "Main Hall", "Panel: Hiring, growing and firing in an AI-first team"],
            ["12:45 – 13:45", "Foyer", "Lunch"],
            ["13:45 – 15:00", "Hall A", "Open space: attendee-led sessions"],
            ["15:15 – 16:00", "Main Hall", "Closing keynote and farewell — Priya Rao"],
        ]),
        PageBreak(),
        Paragraph("<b>Logistics</b>", s["h2"]),
        Paragraph(
            "<b>Venue address:</b> The Harbour Hotel, 1 Pier Road, Edinburgh "
            "EH6 6QW. The nearest tram stop is Ocean Terminal (4 min walk).",
            s["body"],
        ),
        Spacer(1, 8),
        Paragraph(
            "<b>Check-in:</b> The registration desk opens at 08:45 on Monday. "
            "Please bring a photo ID — your lanyard will be issued on the day.",
            s["body"],
        ),
        Spacer(1, 8),
        Paragraph(
            "<b>Dietary needs:</b> Email logistics@northcoastpm.example by "
            "Friday 8 May.",
            s["body"],
        ),
        Spacer(1, 8),
        Paragraph(
            "<b>Slack:</b> Join #summit-2027 in the NorthCoast community for "
            "live updates and attendee introductions.",
            s["body"],
        ),
        Spacer(1, 18),
        Paragraph(
            "We can't wait to see you in Edinburgh. — The NorthCoast team",
            s["muted"],
        ),
    ]
    doc.build(story)
    print(f"wrote {path}")


def build_flight_itinerary() -> None:
    path = OUT_DIR / "flight_itinerary.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )
    s = _styles()

    def seg_table(rows: list[list[str]]) -> Table:
        return Table(
            rows,
            colWidths=[1.5 * inch, 4.4 * inch],
            style=TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 11),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]),
        )

    story = [
        Paragraph("SkyBridge Travel — Trip itinerary", s["h1"]),
        Paragraph(
            "Booking reference <b>SB-7Q2K9P</b> · Traveller: Ms A. Patel · "
            "Issued 10 November 2026",
            s["muted"],
        ),
        Spacer(1, 18),
        Paragraph("<b>Outbound flight</b>", s["h2"]),
        seg_table([
            ["Flight", "BA 442 — British Airways"],
            ["Date", "Thursday 4 March 2027"],
            ["Depart", "London Heathrow (LHR), Terminal 5 — 09:40"],
            ["Arrive", "Barcelona El Prat (BCN), Terminal 1 — 13:15 (local)"],
            ["Duration", "2h 35m · non-stop"],
            ["Cabin / seat", "Economy · 14C (aisle)"],
            ["Baggage", "1 x 23 kg hold bag included"],
            ["Check-in", "Online check-in opens 24h before departure"],
        ]),
        Spacer(1, 16),
        Paragraph("<b>Hotel</b>", s["h2"]),
        seg_table([
            ["Hotel", "Hotel Marina Bella"],
            ["Address", "Carrer de la Marina 118, 08005 Barcelona, Spain"],
            ["Check in", "Thursday 4 March 2027, from 15:00"],
            ["Check out", "Sunday 7 March 2027, by 11:00"],
            ["Room", "Double room, sea view, breakfast included"],
            ["Confirmation", "HMB-554-2027-A"],
        ]),
        PageBreak(),
        Paragraph("<b>Return flight</b>", s["h2"]),
        seg_table([
            ["Flight", "BA 475 — British Airways"],
            ["Date", "Sunday 7 March 2027"],
            ["Depart", "Barcelona El Prat (BCN), Terminal 1 — 16:20 (local)"],
            ["Arrive", "London Heathrow (LHR), Terminal 5 — 17:55"],
            ["Duration", "2h 35m · non-stop"],
            ["Cabin / seat", "Economy · 11F (window)"],
            ["Baggage", "1 x 23 kg hold bag included"],
        ]),
        Spacer(1, 18),
        Paragraph("<b>Useful information</b>", s["h2"]),
        Paragraph(
            "Passport: must be valid for at least 3 months beyond your return "
            "date. UK travellers no longer require a visa for stays under 90 "
            "days in the Schengen area but entry conditions may change — "
            "please check Gov.uk before you travel.",
            s["body"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Transfers: Aerobús from BCN T1 to Plaça de Catalunya runs every "
            "5 minutes (approx 35 min). From there Hotel Marina Bella is a "
            "12-minute taxi.",
            s["body"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Need to change this booking? Call SkyBridge on 0330 555 0199 or "
            "reply to your confirmation email. Changes within 48 hours of "
            "departure incur a fee.",
            s["body"],
        ),
    ]
    doc.build(story)
    print(f"wrote {path}")


def main() -> None:
    build_concert_ticket()
    build_dentist_reminder()
    build_conference_agenda()
    build_flight_itinerary()


if __name__ == "__main__":
    main()
