from datetime import date, timedelta
from astral.sun import sun
from astral.geocoder import lookup, database

# from models.model_hackathon.data import ControlParser

start_date = "2021-03-05"
end_date = "2021-03-08"

start = date.fromisoformat("2021-04-05")
end = date.fromisoformat("2021-04-11")

city = lookup("Amsterdam", database())
s = sun(city.observer, date=start, tzinfo=city.timezone)
print("sunrise", s['sunrise'].hour, s['sunrise'].minute)
print("sunset", s['sunset'].hour, s['sunset'].minute)


def get_sun_rise_and_set(dateinfo, cityinfo):
    s = sun(cityinfo.observer, date=dateinfo, tzinfo=cityinfo.timezone)
    h_r = s['sunrise'].hour + s['sunrise'].minute / 60
    h_s = s['sunset'].hour + s['sunset'].minute / 60
    return h_r, h_s

def get_endTime(start_date, end_date):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    num_days = (end - start).days

    data = {}
    for d in range(num_days):
        cur = start + timedelta(days=d)
        key = "{:02d}-{:02d}".format(cur.day, cur.month)
        # r, s = ControlParser.get_sun_rise_and_set(cur, city)
        r, s = get_sun_rise_and_set(cur, city)
        data[key] = s

    return data


def get_hoursLight(start_date, end_date):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    num_days = (end - start).days

    data = {}
    for d in range(num_days):
        cur = start + timedelta(days=d)
        key = "{:02d}-{:02d}".format(cur.day, cur.month)
        # r, s = ControlParser.get_sun_rise_and_set(cur, city)
        r, s = get_sun_rise_and_set(cur, city)
        data[key] = s - r

    return data


print(get_endTime(start_date, end_date))
print(get_hoursLight(start_date, end_date))