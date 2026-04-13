"""
generate_contracts.py — Generates synthetic legal contracts with automatic annotations.

Produces realistic short contracts covering:
- Service Agreements
- Non-Disclosure Agreements (NDAs)
- Employment Contracts
- Lease Agreements
- Consultancy Agreements

All contracts are auto-annotated with DATE, PARTY, MONEY, TERMINATION labels
and saved directly to data/annotations/train.jsonl

Usage:
    poetry run python generate_contracts.py

No API calls — runs entirely locally.
"""

import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_FILE = Path("data/annotations/train.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Data pools ────────────────────────────────────────────────────────────────

COMPANY_NAMES = [
    "Nexus Digital Solutions Pvt Ltd",
    "Horizon Tech Innovations LLC",
    "Pinnacle Software Services Ltd",
    "Quantum Analytics Corp",
    "Vertex Systems Inc",
    "Aurora Consulting Group LLP",
    "Meridian Global Services Ltd",
    "Zenith Technologies Pvt Ltd",
    "Apex Business Solutions LLC",
    "Catalyst IT Services Corp",
    "Luminary Designs Ltd",
    "Stellar Communications Inc",
    "Vanguard Enterprises LLP",
    "Summit Advisory Services Ltd",
    "Crestview Analytics Pvt Ltd",
    "Pathway Digital Inc",
    "Ironclad Ventures LLC",
    "Skyline Solutions Corp",
    "Bluewave Technologies Ltd",
    "Redstone Consulting Pvt Ltd",
    "Greenfield Software LLC",
    "Cloudbase Systems Inc",
    "Forefront Analytics Ltd",
    "Brightside Services Corp",
    "Eastgate Technologies LLP",
]

INDIVIDUAL_NAMES = [
    "James Mitchell", "Sarah Thompson", "David Patel",
    "Emily Rodriguez", "Michael Chen", "Jessica Williams",
    "Robert Kumar", "Amanda Foster", "Christopher Lee",
    "Priya Sharma", "Daniel O'Brien", "Laura Martinez",
]

AMOUNTS = [
    ("$10,000", 10000, "USD"),
    ("$25,000", 25000, "USD"),
    ("$50,000", 50000, "USD"),
    ("$75,000", 75000, "USD"),
    ("$100,000", 100000, "USD"),
    ("$150,000", 150000, "USD"),
    ("$200,000", 200000, "USD"),
    ("$5,000", 5000, "USD"),
    ("$15,000", 15000, "USD"),
    ("$30,000", 30000, "USD"),
    ("$500,000", 500000, "USD"),
    ("$1,000,000", 1000000, "USD"),
    ("£20,000", 20000, "GBP"),
    ("€45,000", 45000, "EUR"),
    ("₹500,000", 500000, "INR"),
]

NOTICE_PERIODS = ["30", "60", "90", "15", "45"]

GOVERNING_LAWS = [
    "State of California", "State of New York", "State of Delaware",
    "England and Wales", "State of Texas", "Province of Ontario",
]


def random_date(start_year=2022, end_year=2025):
    """Generate a random date string."""
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    year = random.randint(start_year, end_year)
    month_idx = random.randint(0, 11)
    day = random.randint(1, days_in_month[month_idx])
    return f"{day:02d} {months[month_idx]} {year}"


def add_months(date_str, months):
    """Add months to a date string and return new date string."""
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    parts = date_str.split()
    day = int(parts[0])
    month_idx = month_names.index(parts[1])
    year = int(parts[2])

    total_months = month_idx + months
    year += total_months // 12
    month_idx = total_months % 12
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(day, days_in_month[month_idx])
    return f"{day:02d} {month_names[month_idx]} {year}"


def find_span(text, phrase):
    """Find start/end of phrase in text."""
    idx = text.find(phrase)
    if idx != -1:
        return idx, idx + len(phrase)
    return None, None


def annotate(text, spans):
    """Build annotation record from text and list of (phrase, label) tuples."""
    labels = []
    seen_spans = set()
    for phrase, label in spans:
        start, end = find_span(text, phrase)
        if start is not None and (start, end) not in seen_spans:
            labels.append([start, end, label])
            seen_spans.add((start, end))
    labels.sort(key=lambda x: x[0])
    return {"text": text, "label": labels}


# ── Contract Templates ────────────────────────────────────────────────────────

def generate_service_agreement():
    party_a = random.choice(COMPANY_NAMES)
    party_b = random.choice([c for c in COMPANY_NAMES if c != party_a])
    sign_date = random_date()
    start_date = add_months(sign_date, 1)
    end_date = add_months(start_date, random.choice([6, 12, 24]))
    amount = random.choice(AMOUNTS)
    notice = random.choice(NOTICE_PERIODS)
    law = random.choice(GOVERNING_LAWS)
    installments = random.choice(["two equal installments", "four quarterly installments", "monthly installments"])

    text = f"""SERVICE AGREEMENT

This Service Agreement (the "Agreement") is entered into as of {sign_date} (the "Effective Date"), by and between {party_a} (hereinafter referred to as 'Party A'), a company duly incorporated under applicable laws, and {party_b} (hereinafter referred to as 'Party B').

1. Scope of Services
Party B agrees to provide software development and technology consulting services to Party A as described in Schedule A attached hereto. Party B shall perform all services in a professional and workmanlike manner.

2. Payment Terms
In consideration for the services rendered, Party A agrees to pay Party B a total contract value of {amount[0]} ({amount[0]} only), payable in {installments}. The first installment shall be due within fifteen (15) days of the execution of this Agreement. All payments shall be made by bank transfer to the account designated by Party B.

3. Term
This Agreement shall commence on {start_date} and shall continue in full force and effect until {end_date}, unless earlier terminated in accordance with the provisions hereof.

4. Termination
Either party may terminate this Agreement for convenience upon {notice} days written notice to the other party. Either party may terminate this Agreement for cause immediately upon written notice if the other party commits a material breach of this Agreement and fails to cure such breach within fifteen (15) days of receiving written notice thereof.

5. Confidentiality
Each party agrees to keep confidential all proprietary information disclosed by the other party during the term of this Agreement and for a period of three (3) years thereafter.

6. Governing Law
This Agreement shall be governed by and construed in accordance with the laws of the {law}, without regard to its conflict of law provisions.

7. Entire Agreement
This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof and supersedes all prior agreements, understandings, negotiations and discussions.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

{party_a}                                    {party_b}
Authorized Signatory                          Authorized Signatory"""

    spans = [
        (sign_date, "DATE"),
        (start_date, "DATE"),
        (end_date, "DATE"),
        (party_a, "PARTY"),
        (party_b, "PARTY"),
        (amount[0], "MONEY"),
        (f"Either party may terminate this Agreement for convenience upon {notice} days written notice to the other party.", "TERMINATION"),
        (f"Either party may terminate this Agreement for cause immediately upon written notice if the other party commits a material breach of this Agreement and fails to cure such breach within fifteen (15) days of receiving written notice thereof.", "TERMINATION"),
    ]
    return annotate(text, spans)


def generate_nda():
    party_a = random.choice(COMPANY_NAMES)
    party_b = random.choice([c for c in COMPANY_NAMES if c != party_a])
    sign_date = random_date()
    end_date = add_months(sign_date, random.choice([12, 24, 36]))
    notice = random.choice(NOTICE_PERIODS)
    penalty = random.choice(AMOUNTS)

    text = f"""NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement (the "Agreement") is made and entered into as of {sign_date}, by and between {party_a} ("Disclosing Party") and {party_b} ("Receiving Party").

WHEREAS, the parties wish to explore a potential business relationship and may disclose certain confidential information to each other;

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

1. Definition of Confidential Information
"Confidential Information" means any information disclosed by either party to the other party, either directly or indirectly, in writing, orally or by inspection of tangible objects, that is designated as confidential or that reasonably should be understood to be confidential given the nature of the information.

2. Obligations
The Receiving Party agrees to: (a) hold the Confidential Information in strict confidence; (b) not disclose the Confidential Information to any third parties without prior written consent; (c) use the Confidential Information solely for the purpose of evaluating the potential business relationship.

3. Term
This Agreement shall remain in effect from {sign_date} until {end_date}, unless earlier terminated by either party upon {notice} days prior written notice.

4. Liquidated Damages
In the event of a breach of this Agreement, the breaching party shall pay liquidated damages of {penalty[0]} to the non-breaching party, which amount represents a reasonable estimate of the damages that would result from such breach.

5. Termination
Either party may terminate this Agreement upon {notice} days written notice to the other party. Upon termination, the Receiving Party shall promptly return or destroy all Confidential Information.

6. Remedies
The parties acknowledge that any breach of this Agreement may cause irreparable harm for which monetary damages would be inadequate.

IN WITNESS WHEREOF, the parties have executed this Non-Disclosure Agreement as of the date first written above."""

    spans = [
        (sign_date, "DATE"),
        (end_date, "DATE"),
        (party_a, "PARTY"),
        (party_b, "PARTY"),
        (penalty[0], "MONEY"),
        (f"Either party may terminate this Agreement upon {notice} days written notice to the other party.", "TERMINATION"),
        (f"Upon termination, the Receiving Party shall promptly return or destroy all Confidential Information.", "TERMINATION"),
    ]
    return annotate(text, spans)


def generate_employment_contract():
    employer = random.choice(COMPANY_NAMES)
    employee = random.choice(INDIVIDUAL_NAMES)
    sign_date = random_date()
    start_date = add_months(sign_date, 0)
    end_date = add_months(start_date, random.choice([12, 24]))
    salary = random.choice(AMOUNTS)
    bonus = random.choice(AMOUNTS)
    notice = random.choice(NOTICE_PERIODS)
    role = random.choice(["Senior Software Engineer", "Product Manager", "Data Scientist",
                          "Business Analyst", "Project Manager", "UX Designer"])

    text = f"""EMPLOYMENT AGREEMENT

This Employment Agreement (the "Agreement") is entered into as of {sign_date}, between {employer} (the "Employer") and {employee} (the "Employee").

1. Position and Duties
The Employer hereby employs the Employee as {role}. The Employee shall perform such duties and responsibilities as are customarily associated with such position and as may be assigned by the Employer from time to time.

2. Term of Employment
The employment shall commence on {start_date} and shall continue until {end_date}, subject to earlier termination as provided herein.

3. Compensation
3.1 Base Salary: The Employee shall receive an annual base salary of {salary[0]}, payable in equal bi-monthly installments, subject to applicable tax withholdings.
3.2 Performance Bonus: The Employee shall be eligible for an annual performance bonus of up to {bonus[0]}, based on the achievement of mutually agreed performance targets.

4. Termination
4.1 Termination by Employer: The Employer may terminate this Agreement without cause upon {notice} days written notice to the Employee.
4.2 Termination by Employee: The Employee may terminate this Agreement upon {notice} days written notice to the Employer.
4.3 Termination for Cause: The Employer may terminate this Agreement immediately for cause, including but not limited to gross misconduct, willful neglect of duties, or material breach of this Agreement.

5. Confidentiality
The Employee agrees to maintain the confidentiality of all proprietary and confidential information of the Employer during and after the term of employment.

6. Non-Compete
During the term of employment and for a period of one (1) year thereafter, the Employee shall not engage in any business activity that directly competes with the Employer.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above."""

    spans = [
        (sign_date, "DATE"),
        (start_date, "DATE"),
        (end_date, "DATE"),
        (employer, "PARTY"),
        (employee, "PARTY"),
        (salary[0], "MONEY"),
        (bonus[0], "MONEY"),
        (f"The Employer may terminate this Agreement without cause upon {notice} days written notice to the Employee.", "TERMINATION"),
        (f"The Employee may terminate this Agreement upon {notice} days written notice to the Employer.", "TERMINATION"),
        (f"The Employer may terminate this Agreement immediately for cause, including but not limited to gross misconduct, willful neglect of duties, or material breach of this Agreement.", "TERMINATION"),
    ]
    return annotate(text, spans)


def generate_consultancy_agreement():
    client = random.choice(COMPANY_NAMES)
    consultant = random.choice([c for c in COMPANY_NAMES if c != client])
    sign_date = random_date()
    start_date = add_months(sign_date, 0)
    end_date = add_months(start_date, random.choice([3, 6, 12]))
    monthly_fee = random.choice(AMOUNTS)
    total_fee = random.choice(AMOUNTS)
    notice = random.choice(NOTICE_PERIODS)
    domain = random.choice(["management consulting", "technology advisory",
                            "financial consulting", "legal consulting",
                            "marketing strategy", "HR consulting"])

    text = f"""CONSULTANCY AGREEMENT

This Consultancy Agreement (the "Agreement") is made as of {sign_date} between {client} (the "Client") and {consultant} (the "Consultant").

1. Services
The Consultant agrees to provide {domain} services to the Client as mutually agreed upon from time to time. The Consultant shall devote sufficient time and attention to the performance of the services.

2. Fees and Payment
2.1 The Client shall pay the Consultant a monthly retainer fee of {monthly_fee[0]}, due and payable on the first business day of each month.
2.2 The total fees payable under this Agreement shall not exceed {total_fee[0]} without prior written approval from the Client.
2.3 The Consultant shall submit monthly invoices detailing the services rendered.

3. Term
This Agreement shall commence on {start_date} and expire on {end_date}, unless earlier terminated as provided herein.

4. Termination
4.1 Either party may terminate this Agreement for convenience upon {notice} days prior written notice to the other party.
4.2 The Client may terminate this Agreement immediately upon written notice if the Consultant fails to perform the services in accordance with the standards set forth herein.
4.3 This Agreement shall terminate automatically upon the insolvency or bankruptcy of either party.

5. Independent Contractor
The Consultant is an independent contractor and not an employee of the Client. The Consultant shall be responsible for all taxes and insurance related to the services provided.

6. Intellectual Property
All work product, deliverables, and intellectual property created by the Consultant in connection with the services shall be the exclusive property of the Client.

7. Dispute Resolution
Any disputes arising out of or in connection with this Agreement shall be resolved through binding arbitration.

IN WITNESS WHEREOF, the parties have duly executed this Consultancy Agreement as of the date first above written."""

    spans = [
        (sign_date, "DATE"),
        (start_date, "DATE"),
        (end_date, "DATE"),
        (client, "PARTY"),
        (consultant, "PARTY"),
        (monthly_fee[0], "MONEY"),
        (total_fee[0], "MONEY"),
        (f"Either party may terminate this Agreement for convenience upon {notice} days prior written notice to the other party.", "TERMINATION"),
        (f"The Client may terminate this Agreement immediately upon written notice if the Consultant fails to perform the services in accordance with the standards set forth herein.", "TERMINATION"),
        (f"This Agreement shall terminate automatically upon the insolvency or bankruptcy of either party.", "TERMINATION"),
    ]
    return annotate(text, spans)


def generate_lease_agreement():
    landlord = random.choice(COMPANY_NAMES)
    tenant = random.choice([c for c in COMPANY_NAMES if c != landlord])
    sign_date = random_date()
    start_date = add_months(sign_date, 1)
    end_date = add_months(start_date, random.choice([12, 24, 36]))
    monthly_rent = random.choice(AMOUNTS)
    deposit = random.choice(AMOUNTS)
    notice = random.choice(NOTICE_PERIODS)
    address = random.choice([
        "Suite 400, 123 Business Park, New York, NY 10001",
        "Floor 7, Tower B, 456 Corporate Avenue, Los Angeles, CA 90001",
        "Unit 205, Tech Hub, 789 Innovation Drive, San Francisco, CA 94105",
        "Level 3, Skyview Complex, 321 Commerce Street, Chicago, IL 60601",
    ])

    text = f"""COMMERCIAL LEASE AGREEMENT

This Commercial Lease Agreement (the "Lease") is entered into as of {sign_date}, between {landlord} (the "Landlord") and {tenant} (the "Tenant").

1. Premises
The Landlord hereby leases to the Tenant the commercial premises located at {address} (the "Premises"), for use as office space.

2. Term
The lease term shall commence on {start_date} and expire on {end_date} (the "Term"), unless sooner terminated as provided herein.

3. Rent
3.1 The Tenant shall pay monthly rent of {monthly_rent[0]}, due and payable on the first day of each calendar month.
3.2 Security Deposit: Upon execution of this Lease, the Tenant shall deposit {deposit[0]} as a security deposit, to be held by the Landlord throughout the Term.
3.3 All rent payments shall be made by electronic transfer to the Landlord's designated bank account.

4. Termination
4.1 Either party may terminate this Lease upon {notice} days written notice to the other party.
4.2 The Landlord may terminate this Lease immediately upon written notice if the Tenant fails to pay rent within ten (10) days of the due date.
4.3 The Tenant may terminate this Lease without cause upon {notice} days prior written notice, subject to payment of all outstanding rent and charges.

5. Maintenance and Repairs
The Tenant shall maintain the Premises in good condition and repair, reasonable wear and tear excepted. The Landlord shall be responsible for structural repairs and maintenance of common areas.

6. Alterations
The Tenant shall not make any alterations to the Premises without the prior written consent of the Landlord.

7. Insurance
The Tenant shall maintain comprehensive general liability insurance with a minimum coverage of {deposit[0]} throughout the Term of this Lease.

IN WITNESS WHEREOF, the parties have executed this Commercial Lease Agreement as of the date first written above."""

    spans = [
        (sign_date, "DATE"),
        (start_date, "DATE"),
        (end_date, "DATE"),
        (landlord, "PARTY"),
        (tenant, "PARTY"),
        (monthly_rent[0], "MONEY"),
        (deposit[0], "MONEY"),
        (f"Either party may terminate this Lease upon {notice} days written notice to the other party.", "TERMINATION"),
        (f"The Landlord may terminate this Lease immediately upon written notice if the Tenant fails to pay rent within ten (10) days of the due date.", "TERMINATION"),
        (f"The Tenant may terminate this Lease without cause upon {notice} days prior written notice, subject to payment of all outstanding rent and charges.", "TERMINATION"),
    ]
    return annotate(text, spans)


# ── Main ──────────────────────────────────────────────────────────────────────

GENERATORS = [
    generate_service_agreement,
    generate_nda,
    generate_employment_contract,
    generate_consultancy_agreement,
    generate_lease_agreement,
]

def main():
    random.seed(42)
    count = 0
    target = 100  # Generate 100 synthetic contracts

    print(f"\nGenerating {target} synthetic legal contracts...")
    print(f"Output: {OUTPUT_FILE}\n")

    label_counts = {"DATE": 0, "PARTY": 0, "MONEY": 0, "TERMINATION": 0}

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for i in range(target):
            generator = random.choice(GENERATORS)
            record = generator()

            # Verify all spans are valid
            valid = True
            for start, end, label in record["label"]:
                if record["text"][start:end] == "":
                    valid = False
                    break
                label_counts[label] = label_counts.get(label, 0) + 1

            if valid and record["label"]:
                f.write(json.dumps(record) + "\n")
                count += 1

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{target}...")

    print(f"\n✅ Done! Generated {count} contracts.")
    print(f"\nLabel distribution:")
    for label, c in label_counts.items():
        print(f"  {label}: {c}")

    # Count total records in file
    total = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
    print(f"\nTotal records in {OUTPUT_FILE}: {total}")
    print(f"\nNext steps:")
    print(f"  1. Run: poetry run python split_data.py")
    print(f"  2. Run: poetry run python training/train.py --base_model nlpaueb/legal-bert-base-uncased --output_dir models/legal-ner-bert-v3")


if __name__ == "__main__":
    main()