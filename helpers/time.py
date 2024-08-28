def time_to_seconds(time_str):
    parts = list(map(int, time_str.split(":")))

    if len(parts) == 3:  # hh:mm:ss format
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:  # mm:ss format
        minutes, seconds = parts
        return minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format. Use hh:mm:ss or mm:ss.")


if __name__ == "__main__":
    print(time_to_seconds("13:56"))
    print(time_to_seconds("1:42:05"))
