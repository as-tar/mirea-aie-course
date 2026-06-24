import re

PHONE_PATTERN = r"\+?[78][-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}"
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

INN_PATTERN = r"\b\d{10}\b|\b\d{12}\b"

KPP_PATTERN = r"\b\d{9}\b"

class DynamicPIIMasker:
    def __init__(self):
        self.de_mask_map = {}
        self.phone_counter = 0
        self.email_counter = 0
        self.inn_counter = 0
        self.kpp_counter = 0

    def mask(self, text: str) -> str:
        if not text:
            return text
            
        phones = re.findall(PHONE_PATTERN, text)
        for phone in phones:
            if phone not in self.de_mask_map.values():
                placeholder = f"[PHONE_{self.phone_counter}]"
                self.de_mask_map[placeholder] = phone
                self.phone_counter += 1
                text = text.replace(phone, placeholder)
                
        emails = re.findall(EMAIL_PATTERN, text)
        for email in emails:
            if email not in self.de_mask_map.values():
                placeholder = f"[EMAIL_{self.email_counter}]"
                self.de_mask_map[placeholder] = email
                self.email_counter += 1
                text = text.replace(email, placeholder)
                
        inns = re.findall(INN_PATTERN, text)
        for inn in inns:
            if inn not in self.de_mask_map.values():
                placeholder = f"[INN_{self.inn_counter}]"
                self.de_mask_map[placeholder] = inn
                self.inn_counter += 1
                text = text.replace(inn, placeholder)
                
        kpps = re.findall(KPP_PATTERN, text)
        for kpp in kpps:
            if kpp not in self.de_mask_map.values():
                placeholder = f"[KPP_{self.kpp_counter}]"
                self.de_mask_map[placeholder] = kpp
                self.kpp_counter += 1
                text = text.replace(kpp, placeholder)
                
        return text

    def unmask(self, text: str) -> str:
        if not text:
            return text
        for placeholder, original in self.de_mask_map.items():
            text = text.replace(placeholder, original)
        return text