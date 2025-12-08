import os
# токен профиля для Тинькофф Инвестиций
# инструкция https://tinkoff.github.io/investAPI/token/

INVEST_TOKEN="t.rU2bKskjPpnC4HTXn7gxhPegAKvdKFq5_KuB6zvbDDl37WGIJSKnfx6NWp1T13c5V33qZ-Pg3KlRCZ2rjhV3dg"
# INVEST_SANDBOX_TOKEN=os.environ["INVEST_SANDBOX_TOKEN"]

# режим работы: песочница или реальный счет
IS_SANDBOX = True

# массив анализируемых инструментов, которые содержатся в индексе Московской биржи (IMOEX) generate_instruments.py
INSTRUMENTS = [
    { "name": "AFKS", "alias": "АФК Система", "figi": "BBG004S68614", "full_name": "АФК Система", "class_code": "TQBR" },
    { "name": "AFLT", "alias": "Аэрофлот", "figi": "BBG004S683W7", "full_name": "Аэрофлот", "class_code": "TQBR" },
    { "name": "ALRS", "alias": "АЛРОСА", "figi": "BBG004S68B31", "full_name": "АЛРОСА", "class_code": "TQBR" },
    { "name": "ASTR", "alias": "Группа Астра", "figi": "RU000A106T36", "full_name": "Группа Астра", "class_code": "TQBR" },
    { "name": "BSPB", "alias": "Банк Санкт-Петербург", "figi": "BBG000QJW156", "full_name": "Банк Санкт-Петербург", "class_code": "TQBR" },
    { "name": "CBOM", "alias": "МКБ", "figi": "BBG009GSYN76", "full_name": "МКБ", "class_code": "TQBR" },
    { "name": "CHMF", "alias": "Северсталь", "figi": "BBG00475K6C3", "full_name": "Северсталь", "class_code": "TQBR" },
    { "name": "ENPG", "alias": "Эн+", "figi": "BBG000RMWQD4", "full_name": "Эн+", "class_code": "TQBR" },
    { "name": "FEES", "alias": "ФСК Россети", "figi": "BBG00475JZZ6", "full_name": "ФСК Россети", "class_code": "TQBR" },
    { "name": "FLOT", "alias": "Совкомфлот", "figi": "BBG000R04X57", "full_name": "Совкомфлот", "class_code": "TQBR" },
    { "name": "GAZP", "alias": "Газпром", "figi": "BBG004730RP0", "full_name": "Газпром", "class_code": "TQBR" },
    { "name": "GMKN", "alias": "Норильский никель", "figi": "BBG004731489", "full_name": "Норильский никель", "class_code": "TQBR" },
    { "name": "HEAD", "alias": "Хэдхантер", "figi": "TCS20A107662", "full_name": "Хэдхантер", "class_code": "TQBR" },
    { "name": "HYDR", "alias": "РусГидро", "figi": "BBG00475K2X9", "full_name": "РусГидро", "class_code": "TQBR" },
    { "name": "IRAO", "alias": "Интер РАО ЕЭС", "figi": "BBG004S68473", "full_name": "Интер РАО ЕЭС", "class_code": "TQBR" },
    { "name": "LEAS", "alias": "Европлан", "figi": "TCS00A0ZZFS9", "full_name": "Европлан", "class_code": "TQBR" },
    { "name": "LKOH", "alias": "ЛУКОЙЛ", "figi": "BBG004731032", "full_name": "ЛУКОЙЛ", "class_code": "TQBR" },
    { "name": "MAGN", "alias": "Магнитогорский металлургический комбинат", "figi": "BBG004S68507", "full_name": "Магнитогорский металлургический комбинат", "class_code": "TQBR" },
    { "name": "MGNT", "alias": "Магнит", "figi": "BBG004RVFCY3", "full_name": "Магнит", "class_code": "TQBR" },
    { "name": "MOEX", "alias": "Московская Биржа", "figi": "BBG004730JJ5", "full_name": "Московская Биржа", "class_code": "TQBR" },
    { "name": "MSNG", "alias": "Мосэнерго", "figi": "BBG004S687W8", "full_name": "Мосэнерго", "class_code": "TQBR" },
    { "name": "MTLR", "alias": "Мечел", "figi": "BBG004S68598", "full_name": "Мечел", "class_code": "TQBR" },
    { "name": "MTSS", "alias": "МТС", "figi": "BBG004S681W1", "full_name": "МТС", "class_code": "TQBR" },
    { "name": "NLMK", "alias": "НЛМК", "figi": "BBG004S681B4", "full_name": "НЛМК", "class_code": "TQBR" },
    { "name": "NVTK", "alias": "НОВАТЭК", "figi": "BBG00475KKY8", "full_name": "НОВАТЭК", "class_code": "TQBR" },
    { "name": "PHOR", "alias": "ФосАгро", "figi": "BBG004S689R0", "full_name": "ФосАгро", "class_code": "TQBR" },
    { "name": "PIKK", "alias": "ПИК СЗ (ПАО) ао", "figi": "BBG004S68BH6", "full_name": "ПИК СЗ (ПАО) ао", "class_code": "TQBR" },
    { "name": "PLZL", "alias": "Полюс", "figi": "BBG000R607Y3", "full_name": "Полюс", "class_code": "TQBR" },
    { "name": "POSI", "alias": "Группа Позитив", "figi": "TCS00A103X66", "full_name": "Группа Позитив", "class_code": "TQBR" },
    { "name": "ROSN", "alias": "Роснефть", "figi": "BBG004731354", "full_name": "Роснефть", "class_code": "TQBR" },
    { "name": "RTKM", "alias": "Ростелеком", "figi": "BBG004S682Z6", "full_name": "Ростелеком", "class_code": "TQBR" },
    { "name": "RUAL", "alias": "РУСАЛ", "figi": "BBG008F2T3T2", "full_name": "РУСАЛ", "class_code": "TQBR" },
    { "name": "SBER", "alias": "Сбер Банк", "figi": "BBG004730N88", "full_name": "Сбер Банк", "class_code": "TQBR" },
    { "name": "SELG", "alias": "Селигдар", "figi": "BBG002458LF8", "full_name": "Селигдар", "class_code": "TQBR" },
    { "name": "SMLT", "alias": "ГК Самолет", "figi": "BBG00F6NKQX3", "full_name": "ГК Самолет", "class_code": "TQBR" },
    { "name": "SNGS", "alias": "Сургутнефтегаз", "figi": "BBG0047315D0", "full_name": "Сургутнефтегаз", "class_code": "TQBR" },
    { "name": "SVCB", "alias": "Совкомбанк", "figi": "TCS00A0ZZAC4", "full_name": "Совкомбанк", "class_code": "TQBR" },
    { "name": "T", "alias": "Т-Технологии", "figi": "TCS80A107UL4", "full_name": "Т-Технологии", "class_code": "TQBR" },
    { "name": "TATN", "alias": "Татнефть", "figi": "BBG004RVFFC0", "full_name": "Татнефть", "class_code": "TQBR" },
    { "name": "UGLD", "alias": "Южуралзолото ГК", "figi": "TCS00A0JPP37", "full_name": "Южуралзолото ГК", "class_code": "TQBR" },
    { "name": "UPRO", "alias": "Юнипро", "figi": "BBG004S686W0", "full_name": "Юнипро", "class_code": "TQBR" },
    { "name": "VKCO", "alias": "ВК", "figi": "TCS00A106YF0", "full_name": "ВК", "class_code": "TQBR" },
    { "name": "VTBR", "alias": "Банк ВТБ", "figi": "BBG004730ZJ9", "full_name": "Банк ВТБ", "class_code": "TQBR" },
    { "name": "YDEX", "alias": "Яндекс", "figi": "TCS00A107T19", "full_name": "Яндекс", "class_code": "TQBR" },
]