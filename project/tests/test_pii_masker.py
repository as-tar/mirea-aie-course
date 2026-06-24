import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pytest
from src.utils.pii_masker import DynamicPIIMasker

def test_pii_masking_and_unmasking():
    masker = DynamicPIIMasker()
    
    raw_text = (
        "Контакты поддержки ООО «НейроСети»: email support@neuro-networks.ru, "
        "тел. 8 (999) 888-77-66. ИНН компании: 7712345678, КПП: 771201001."
    )
    
    masked_text = masker.mask(raw_text)
    
    assert "[EMAIL_0]" in masked_text
    assert "[PHONE_0]" in masked_text
    assert "[INN_0]" in masked_text
    assert "[KPP_0]" in masked_text
    
    assert "support@neuro-networks.ru" not in masked_text
    assert "8 (999) 888-77-66" not in masked_text
    assert "7712345678" not in masked_text
    assert "771201001" not in masked_text
    
    unmasked_text = masker.unmask(masked_text)
    assert unmasked_text == raw_text

def test_pii_masker_duplicate_values():
    masker = DynamicPIIMasker()
    text = "Звоните на 89998887766 или на 89998887766."
    
    masked = masker.mask(text)
    
    assert masked.count("[PHONE_0]") == 2
    assert "[PHONE_1]" not in masked