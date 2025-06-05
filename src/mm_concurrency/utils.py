from datetime import UTC, datetime, timedelta


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_delta(
    *,
    days: int | None = None,
    hours: int | None = None,
    minutes: int | None = None,
    seconds: int | None = None,
) -> datetime:
    params = {}
    if days:
        params["days"] = days
    if hours:
        params["hours"] = hours
    if minutes:
        params["minutes"] = minutes
    if seconds:
        params["seconds"] = seconds
    return datetime.now(UTC) + timedelta(**params)


def is_too_old(value: datetime | None, seconds: int) -> bool:
    return value is None or value < utc_delta(seconds=-1 * seconds)
