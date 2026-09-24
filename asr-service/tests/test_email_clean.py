from pathlib import Path

from asr_service.email_clean import clean_email

DEMO = Path(__file__).resolve().parents[2] / "data" / "cases" / "demo" / "email.eml"


def message(body: str) -> bytes:
    head = "From: Клиент <c@example.com>\nTo: s@example.com\nSubject: t\nContent-Type: text/plain; charset=UTF-8\n\n"
    return (head + body).encode()


def test_demo_email_keeps_only_the_new_text():
    mail = clean_email(DEMO.read_bytes())
    assert mail.sender == "support@kvarta.example"
    assert mail.sender_name == "Служба поддержки «Кварта»"
    assert mail.subject == "Re: Заявка № 290516"
    assert mail.date == "2026-09-15T11:40:12+03:00"
    assert mail.body.startswith("Здравствуйте, Сергей Викторович!")
    assert mail.body.endswith("Приносим извинения за доставленные неудобства.")
    assert "Жду решения" in mail.removed["quoted"]
    assert mail.removed["signature"].startswith("С уважением")


def test_russian_outlook_header_cuts_history():
    body = "Спасибо, получил.\n\nОт: Поддержка <s@example.com>\nОтправлено: 3 сентября 2026 г. 12:00\nТема: заявка\n\nстарый текст"
    mail = clean_email(message(body))
    assert mail.body == "Спасибо, получил."
    assert "старый текст" in mail.removed["quoted"]


def test_original_message_separator():
    mail = clean_email(message("Новый текст\n-----Исходное сообщение-----\nстарый"))
    assert mail.body == "Новый текст"


def test_plain_message_is_untouched():
    mail = clean_email(message("Здравствуйте.\nВопрос по счёту за август."))
    assert mail.body == "Здравствуйте.\nВопрос по счёту за август."
    assert mail.removed == {}
