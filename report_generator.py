import os
import time

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas


# Generate Excel and PDF reports from event history.
def generate_reports(event_history, cheating_score, reports_dir="reports"):
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    excel_path = os.path.join(reports_dir, f"examshield_report_{timestamp}.xlsx")
    pdf_path = os.path.join(reports_dir, f"examshield_report_{timestamp}.pdf")

    # Build the Excel sheet.
    history_rows = []
    for record in event_history:
        history_rows.append({
            "Date & Time": record.get("timestamp", ""),
            "Event": record.get("event", ""),
            "Screenshot Path": record.get("screenshot_path", ""),
            "Cheating Score": record.get("cheating_score", 0),
        })

    reports_df = pd.DataFrame(history_rows)
    if reports_df.empty:
        reports_df = pd.DataFrame([
            {
                "Date & Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Event": "No suspicious events captured",
                "Screenshot Path": "N/A",
                "Cheating Score": cheating_score,
            }
        ])

    reports_df.to_excel(excel_path, index=False, engine="openpyxl")

    # Build the PDF report.
    pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    pdf_canvas.setTitle("ExamShield AI Proctoring Report")
    pdf_canvas.setFont("Helvetica-Bold", 20)
    pdf_canvas.drawString(40, height - 50, "ExamShield AI - Proctoring Report")
    pdf_canvas.setFont("Helvetica", 10)
    pdf_canvas.drawString(40, height - 70, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    pdf_canvas.drawString(40, height - 85, f"Final Cheating Score: {cheating_score}%")

    data = [["Date & Time", "Event", "Screenshot Path", "Cheating Score"]]
    for record in event_history:
        data.append([
            record.get("timestamp", ""),
            record.get("event", ""),
            record.get("screenshot_path", "N/A"),
            str(record.get("cheating_score", 0)),
        ])

    table = Table(data, colWidths=[120, 180, 180, 60])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))

    table.wrapOn(pdf_canvas, width - 80, height - 120)
    table.drawOn(pdf_canvas, 40, height - 120 - (len(data) * 18))

    pdf_canvas.save()
    return excel_path, pdf_path
