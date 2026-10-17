#!/usr/bin/env python3
"""Parity cases: Godwit questions about states.

    cases.py testdata/cases.json                 synthetic mail, safe to commit
    cases.py --states STATES.jsonl OUT.json      real mail (#64's states.jsonl): keep it local

Every state gets Godwit's seven preset label questions, as the label
classifier asks them; the synthetic set adds choice and score questions and
states sized for each context tier.
"""
import argparse
import json
from pathlib import Path

# Godwit's Chinese label presets (core/labels.go) and the classifier's wording (core/labeler.go).
LABELS = [
    ("otp", "验证码", "登录、注册、支付时收到的一次性验证码或登录链接"),
    ("bill", "账单", "账单、收据、发票和扣款通知"),
    ("order", "订单与物流", "网购订单确认、发货和快递进度"),
    ("travel", "航班与出行", "机票、火车票、酒店预订和值机提醒"),
    ("calendar", "日程邀请", "会议邀请、日程变更和活动报名确认"),
    ("notice", "通知", "系统或服务自动发出的提醒，不需要回复"),
    ("newsletter", "订阅与资讯", "新闻简报、产品更新和营销邮件"),
]


def label_questions():
    return [{"name": slug, "type": "predicate",
             "instructions": f"Answer yes if this email belongs to the user's label \"{name}\". {desc}".strip()}
            for slug, name, desc in LABELS]


def mail(n, subject, sender, body):
    text = f"Subject: {subject}\nFrom: {sender}\nDate: 2026-10-01T09:00:00Z\n\n{body}"
    return f'<untrusted_email source="message:{n}">\n{text}\n</untrusted_email>'


PARAGRAPH = ("Thank you for your order. We have received your payment of 42.00 USD and will ship the items within two "
             "business days. You can follow the delivery on our website with the tracking number below. ")

SYNTHETIC = [
    ("bill-en", mail(1, "Your October invoice", "Acme Cloud <billing@acme.example>",
                     "Your invoice for October is ready. Amount due: 42.00 USD, charged to the card ending 4242 on Nov 1.")),
    ("otp-zh", mail(2, "您的登录验证码", "某银行 <noreply@bank.example>", "您的验证码是 384920，5 分钟内有效。请勿告诉他人。")),
    ("empty", ""),
    ("tier-2048", mail(3, "Order 1024 shipped", "Shop <orders@shop.example>", PARAGRAPH * 22)),
    ("tier-4096", mail(4, "Order 2048 shipped", "Shop <orders@shop.example>", PARAGRAPH * 50)),
]

EXTRA = [
    {"name": "kind", "type": "choice", "instructions": "What kind of email is this?",
     "choices": [{"value": "personal", "description": "Written by a person to the user"},
                 {"value": "automated", "description": "Sent by a system or service"},
                 {"value": "marketing", "description": ""}]},
    {"name": "bare", "type": "choice", "instructions": "Pick the best summary.",
     "choices": [{"value": "A", "description": "A payment request"}, {"value": "B", "description": "A delivery update"},
                 {"value": "C", "description": "A sign-in code"}]},
    {"name": "same", "type": "choice", "instructions": "Which department should read it?",
     "choices": [{"value": "billing", "description": "Billing"}, {"value": "support", "description": "support"}]},
    {"name": "urgency", "type": "score", "instructions": "How urgent is this email?",
     "levels": [{"label": "none", "description": "Can be ignored"}, {"label": "low", "description": ""},
                {"label": "", "description": "Should be read today"}, {"label": "high", "description": "Act now"}]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    cases = []
    if args.states:
        for line in args.states.read_text().splitlines():
            r = json.loads(line)
            cases += [{"name": f"{r['id']}-{q['name']}", "state": r["state"], "question": q} for q in label_questions()]
    else:
        for name, state in SYNTHETIC:
            cases += [{"name": f"{name}-{q['name']}", "state": state, "question": q} for q in label_questions()]
            if name in ("bill-en", "empty"):
                cases += [{"name": f"{name}-{q['name']}", "state": state, "question": q} for q in EXTRA]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
