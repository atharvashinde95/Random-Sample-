EMPLOYEES = {
    "EMP001": {
        "name": "Atharva Sharma",
        "department": "Engineering",
        "email": "atharva.sharma@company.com",
        "manager": "Rahul Gupta",
        "joining_date": "2022-03-15",
        "location": "Mumbai",
        "grade": "L3",
    },
    "EMP002": {
        "name": "Priya Patel",
        "department": "Human Resources",
        "email": "priya.patel@company.com",
        "manager": "Sunita Mehta",
        "joining_date": "2021-07-01",
        "location": "Pune",
        "grade": "L4",
    },
    "EMP003": {
        "name": "Rohan Desai",
        "department": "Finance",
        "email": "rohan.desai@company.com",
        "manager": "Amit Shah",
        "joining_date": "2023-01-10",
        "location": "Bangalore",
        "grade": "L2",
    },
}

LEAVE_BALANCES = {
    "EMP001": {"casual": 8,  "sick": 5,  "earned": 12, "total_used_this_year": 7},
    "EMP002": {"casual": 3,  "sick": 10, "earned": 18, "total_used_this_year": 14},
    "EMP003": {"casual": 10, "sick": 7,  "earned": 5,  "total_used_this_year": 3},
}

TICKETS = {
    "EMP001": [
        {
            "id": "TKT-2449", "title": "Laptop overheating issue",
            "status": "Resolved", "created": "2025-03-10", "updated": "2025-03-14",
            "priority": "High", "category": "Hardware",
            "description": "Laptop was overheating during video calls. RAM upgraded and thermal paste replaced.",
        },
        {
            "id": "TKT-2451", "title": "VPN access not working from home",
            "status": "In Progress", "created": "2025-03-28", "updated": "2025-04-02",
            "priority": "Medium", "category": "Network",
            "description": "Unable to connect to corporate VPN from home. IT team investigating firewall rules.",
        },
        {
            "id": "TKT-2455", "title": "IntelliJ IDEA license renewal",
            "status": "Open", "created": "2025-04-01", "updated": "2025-04-01",
            "priority": "Low", "category": "Software",
            "description": "IntelliJ license expires April 30. Renewal submitted to procurement.",
        },
    ],
    "EMP002": [
        {
            "id": "TKT-2430", "title": "Payroll discrepancy - March",
            "status": "Resolved", "created": "2025-02-20", "updated": "2025-03-01",
            "priority": "High", "category": "Payroll",
            "description": "Incorrect HRA in March payslip. Finance corrected and reissued.",
        },
        {
            "id": "TKT-2444", "title": "New hire onboarding kit request",
            "status": "Closed", "created": "2025-03-05", "updated": "2025-03-08",
            "priority": "Medium", "category": "Onboarding",
            "description": "Onboarding kits for 3 new HR joiners. All dispatched.",
        },
    ],
    "EMP003": [
        {
            "id": "TKT-2460", "title": "Travel expense reimbursement pending",
            "status": "In Progress", "created": "2025-04-02", "updated": "2025-04-03",
            "priority": "Medium", "category": "Finance",
            "description": "Travel claim for Bangalore→Mumbai visit on March 28. Under review.",
        },
    ],
}

ALL_TICKETS = {}
for _emp_tickets in TICKETS.values():
    for _t in _emp_tickets:
        ALL_TICKETS[_t["id"]] = _t
