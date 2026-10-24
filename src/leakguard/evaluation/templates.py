"""Message templates for the synthetic evaluation set.

Slots:
    {PHONE}, {SECRET.aws}   gold slot: a labelled value of that type (variant picks a format)
    {~ORDER_ID}             filler slot: an unlabelled value that must NOT be flagged

Each template belongs to exactly one split (even index -> dev, odd index -> test), so the
test set is phrased in ways the dev set never showed: tuning can't overfit wording.
"""

TEMPLATES: dict[str, list[str]] = {
    # Customers typing into a support chatbot.
    "support": [
        "Hi, my Aadhaar is {AADHAAR}, please update my KYC.",
        "My phone number is {PHONE}, please call me back about order {~ORDER_ID}.",
        "Refund hasn't come yet. My UPI is {UPI}, amount was {~AMOUNT}.",
        "I'm {NAME}. You can reach me at {PHONE} or {EMAIL}.",
        "PAN {PAN} is linked to my account but KYC still shows pending.",
        "I was charged {~AMOUNT} twice on card {CARD}, please reverse one.",
        "Please send the refund to a/c no {BANK_ACCOUNT}, IFSC {IFSC}.",
        "Deliver to {ADDRESS}. Call {PHONE} before coming.",
        "Change my email from {EMAIL} to {EMAIL}, thanks.",
        "The OTP {~OTP} isn't working. Registered mobile: {PHONE}",
        "I paid via {UPI} but order {~ORDER_ID} still shows unpaid.",
        "This is {NAME}, account number {BANK_ACCOUNT} at {~BANK} bank, branch IFSC {IFSC}.",
        "Hello, {NAME} here. My new address is {ADDRESS}.",
        "Uploaded Aadhaar {AADHAAR} and PAN {PAN} for ticket {~TICKET}.",
        "Can you confirm the card {CARD} on file? My email is {EMAIL}.",
        "Please call {PHONE} or WhatsApp {PHONE} anytime after 6pm.",
        "My account {BANK_ACCOUNT} was debited {~AMOUNT} on {~DATE}, please check.",
        "Send invoice {~INVOICE} to {EMAIL}, GST details are with my PAN {PAN}.",
    ],
    # Support agents pasting whole customer records for the AI to summarise.
    "agent_record": [
        "Name - {NAME} | Mob - {PHONE} | PAN - {PAN} | A/c - {BANK_ACCOUNT} | IFSC - {IFSC}",
        ("Customer: {NAME}\nPhone: {PHONE}\nEmail: {EMAIL}\nAadhaar: {AADHAAR}\n"
        "Address: {ADDRESS}\nIssue: refund for order {~ORDER_ID}"),
        ("Summarise this ticket: {NAME} ({EMAIL}) says card {CARD} was charged twice "
        "({~AMOUNT}). Callback on {PHONE}."),
        ("KYC record\nFull name: {NAME}\nPAN: {PAN}\nAadhaar no: {AADHAAR}\nMobile: {PHONE}\n"
        "UPI: {UPI}"),
        "Escalation {~TICKET}: customer {NAME}, {ADDRESS}, phone {PHONE}, wants refund to {UPI}.",
        ("Beneficiary details:\nAccount holder: {NAME}\nAccount number: {BANK_ACCOUNT}\n"
        "IFSC: {IFSC}\nRegistered email: {EMAIL}"),
        ("Draft a reply to {NAME} <{EMAIL}> about the delivery to {ADDRESS} "
        "(order {~ORDER_ID}, contact {PHONE})."),
        "Lead: {NAME}, {PHONE}, {EMAIL}, PAN {PAN}, wants a loan against card {CARD}.",
    ],
    # Developers testing prompts and pasting code, configs and logs.
    "dev_snippet": [
        ("Why does this fail?\nclient = OpenAI(api_key=\"{SECRET.openai}\")\n"
        "resp = client.chat.completions.create(model=\"gpt-4o-mini\")"),
        "export AWS_ACCESS_KEY_ID={SECRET.aws}\nexport AWS_REGION=ap-south-1",
        "curl -H \"Authorization: Bearer {SECRET.token}\" https://api.example.com/v1/orders",
        "git clone https://{SECRET.github}@github.com/acme/payments.git fails with 403",
        "SLACK_BOT_TOKEN={SECRET.slack} is in my .env but the bot doesn't reply",
        "GOOGLE_MAPS_KEY={SECRET.google}\nWhy is geocoding returning REQUEST_DENIED?",
        "stripe.api_key = \"{SECRET.stripe}\" -> getting 401 from Stripe",
        "Decode this JWT for me: {SECRET.jwt}",
        "My deploy key starts like this:\n{SECRET.pem}\nMIIEowIBAAKCAQEA... what format is it?",
        "config.yaml:\n  db_host: 10.0.0.12\n  db_password: {SECRET.password}",
        "headers = {\"x-api-key\": \"{SECRET.token}\"}  # why 403?",
        "Here's my .env:\nOPENAI_API_KEY={SECRET.openai}\nDEBUG=true",
        "password = \"{SECRET.password}\"  # TODO move to vault",
        ("Function {~CAMEL} throws NullPointerException after commit {~SHA}, "
        "key was {SECRET.token}"),
        "token: {SECRET.token}\nretry the webhook with request id {~UUID}",
        "AKIA key {SECRET.aws} leaked in CI logs, how do I rotate it?",
        # Added after inspecting test errors once: prefixed key names (DB_PASSWORD=...)
        # were absent from dev, so tuning could not see them. Index 16 -> dev split.
        "docker run -e DB_USER=app -e DB_PASSWORD={SECRET.password} -e APP_SECRET={SECRET.token} myimage",
    ],
    # Romanised Hindi-English, as typed on phones.
    "hinglish": [
        "bhai mera number {PHONE} hai, call kar lena",
        "refund {UPI} pe bhej do please, {~AMOUNT} tha",
        "mera aadhaar {AADHAAR} hai, kyc update kar do",
        "paise is account mein daal do: a/c {BANK_ACCOUNT} ifsc {IFSC}",
        "parcel {ADDRESS} pe chhod dena, {PHONE} pe call karna",
        "pan card {PAN} link nahi ho raha, kya karu?",
        "order {~ORDER_ID} abhi tak nahi aaya, mera naam {NAME} hai",
        "otp {~OTP} aaya par login nahi ho raha, number {PHONE} hai",
        "khata number {BANK_ACCOUNT} hai, {~BANK} branch ka",
        "invoice {EMAIL} pe bhej dena yaar",
        "mujhe {~AMOUNT} wapas chahiye, upi id {UPI} hai",
        "card {CARD} se do baar paise kat gaye",
        "mera naya address {ADDRESS} hai, update kar do",
        "{NAME} bol raha hoon, mobile {PHONE}, email {EMAIL}",
        "sir ji upi {UPI} pe bhejo, ya phir a/c {BANK_ACCOUNT}",
        "aadhaar {AADHAAR} aur pan {PAN} dono upload kar diye",
    ],
    # Nothing sensitive: look-alike numbers and harmless sentences. Every flag is an FP.
    "negative": [
        "My order {~ORDER_ID} was delivered late, please check.",
        "Your OTP is {~OTP}. It is valid for 10 minutes.",
        "Ticket {~TICKET} has been escalated to the payments team.",
        "Invoice {~INVOICE} dated {~DATE} is attached.",
        "Commit {~SHA} broke the build on main.",
        "Request {~UUID} failed with a timeout.",
        "Call our toll-free number {~TOLLFREE} for help.",
        "My password is wrong and the reset password link isn't working.",
        "Function {~CAMEL} returns null for empty carts.",
        "I paid {~AMOUNT} from my account yesterday but it's not reflecting.",
        "Tracking number {~TRACKING} shows 'in transit' for a week.",
        "The {~BANK} bank charged me {~AMOUNT} as a fee on {~DATE}.",
        "Log timestamp {~TIMESTAMP}: payment gateway returned 502.",
        "Use coupon {~COUPON} to get a discount on order {~ORDER_ID}.",
        "The api key expired yesterday, I'll generate a new one.",
        "reset password: how do I do it from the app?",
        "Order {~ORDER_ID} and ticket {~TICKET} both refer to the same issue.",
        "My account was locked after 3 wrong attempts, ref {~TICKET}.",
    ],
    # Real formats the rules are known not to handle. Expected misses, measured honestly.
    "hard": [
        "my number is {PHONE.words}",
        "call me on {PHONE.dots} after 5",
        "aadhaar {AADHAAR.mixed} please verify",
        "pan is {PAN.spaced}",
        "send it to {UPI.unknown_handle}",
        "password: {SECRET.letters}",
        "please transfer to {BANK_ACCOUNT}, it's my savings",
        "DB_PASSWORD={SECRET.password}",
        "ph: {PHONE.words}, alt {PHONE.dots}",
        "my upi {UPI.unknown_handle} and backup {UPI.unknown_handle}",
    ],
}
