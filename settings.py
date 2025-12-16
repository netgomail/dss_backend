import os
from datetime import datetime
# токен профиля для Тинькофф Инвестиций
# инструкция https://tinkoff.github.io/investAPI/token/

INVEST_TOKEN="t.rU2bKskjPpnC4HTXn7gxhPegAKvdKFq5_KuB6zvbDDl37WGIJSKnfx6NWp1T13c5V33qZ-Pg3KlRCZ2rjhV3dg"
# INVEST_SANDBOX_TOKEN=os.environ["INVEST_SANDBOX_TOKEN"]

# режим работы: песочница или реальный счет
IS_SANDBOX = True

# Параметры для загрузки исторических данных с download_md.py
MINIMUM_YEAR = 2009
CURRENT_YEAR = datetime.now().year

# массив анализируемых инструментов, которые содержатся в индексе Московской биржи (IMOEX) generate_instruments.py
INSTRUMENTS = [
    { "ticker": "AFKS", "name": "АФК Система", "figi": "BBG004S68614", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "AFLT", "name": "Аэрофлот", "figi": "BBG004S683W7", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "ALRS", "name": "АЛРОСА", "figi": "BBG004S68B31", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "ASTR", "name": "Группа Астра", "figi": "RU000A106T36", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "BSPB", "name": "Банк Санкт-Петербург", "figi": "BBG000QJW156", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "CBOM", "name": "МКБ", "figi": "BBG009GSYN76", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "CHMF", "name": "Северсталь", "figi": "BBG00475K6C3", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "ENPG", "name": "Эн+", "figi": "BBG000RMWQD4", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "FEES", "name": "ФСК Россети", "figi": "BBG00475JZZ6", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "FLOT", "name": "Совкомфлот", "figi": "BBG000R04X57", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "GAZP", "name": "Газпром", "figi": "BBG004730RP0", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "GMKN", "name": "Норильский никель", "figi": "BBG004731489", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "HEAD", "name": "Хэдхантер", "figi": "TCS20A107662", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "HYDR", "name": "РусГидро", "figi": "BBG00475K2X9", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "IMOEXF", "name": "IMOEXF Индекс МосБиржи", "figi": "FUTIMOEXF000", "class_code": "SPBFUT", "instrument_type": "futures" },
    { "ticker": "IRAO", "name": "Интер РАО ЕЭС", "figi": "BBG004S68473", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "LEAS", "name": "Европлан", "figi": "TCS00A0ZZFS9", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "LKOH", "name": "ЛУКОЙЛ", "figi": "BBG004731032", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "MAGN", "name": "Магнитогорский металлургический комбинат", "figi": "BBG004S68507", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "MGNT", "name": "Магнит", "figi": "BBG004RVFCY3", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "MOEX", "name": "Московская Биржа", "figi": "BBG004730JJ5", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "MSNG", "name": "Мосэнерго", "figi": "BBG004S687W8", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "MTLR", "name": "Мечел", "figi": "BBG004S68598", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "MTSS", "name": "МТС", "figi": "BBG004S681W1", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "NLMK", "name": "НЛМК", "figi": "BBG004S681B4", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "NVTK", "name": "НОВАТЭК", "figi": "BBG00475KKY8", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "PHOR", "name": "ФосАгро", "figi": "BBG004S689R0", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "PIKK", "name": "ПИК СЗ (ПАО) ао", "figi": "BBG004S68BH6", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "PLZL", "name": "Полюс", "figi": "BBG000R607Y3", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "POSI", "name": "Группа Позитив", "figi": "TCS00A103X66", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "ROSN", "name": "Роснефть", "figi": "BBG004731354", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "RTKM", "name": "Ростелеком", "figi": "BBG004S682Z6", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "RUAL", "name": "РУСАЛ", "figi": "BBG008F2T3T2", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "SBER", "name": "Сбер Банк", "figi": "BBG004730N88", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "SELG", "name": "Селигдар", "figi": "BBG002458LF8", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "SMLT", "name": "ГК Самолет", "figi": "BBG00F6NKQX3", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "SNGS", "name": "Сургутнефтегаз", "figi": "BBG0047315D0", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "SVCB", "name": "Совкомбанк", "figi": "TCS00A0ZZAC4", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "T", "name": "Т-Технологии", "figi": "TCS80A107UL4", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "TATN", "name": "Татнефть", "figi": "BBG004RVFFC0", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "UGLD", "name": "Южуралзолото ГК", "figi": "TCS00A0JPP37", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "UPRO", "name": "Юнипро", "figi": "BBG004S686W0", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "VKCO", "name": "ВК", "figi": "TCS00A106YF0", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "VTBR", "name": "Банк ВТБ", "figi": "BBG004730ZJ9", "class_code": "TQBR", "instrument_type": "share" },
    { "ticker": "YDEX", "name": "Яндекс", "figi": "TCS00A107T19", "class_code": "TQBR", "instrument_type": "share" },
]