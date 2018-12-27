import sqlite3
import re


def create_connection():
    try:
        conn = sqlite3.connect('D:\ISIP2018\DBnews.db')
        return conn
    except ValueError as e:
        print(e)
    return None


def remove_non_alphabetic_token(text):
    regex = re.compile('[^a-zA-Z.]')
    return regex.sub(' ', text)


def get_political_news(conn):
    with conn:
        cur = conn.cursor()
        sql = "select news_id, news from t_news where trim(news) " \
              "not like 'detikFlash%' or trim(news) not like 'e-Flash%'"
        cur.execute(sql)
        data = cur.fetchall()
        cur.close()

        return data


def create_per_file(data):
    for file in data:
        if "001012" in file[0]:
            print("pok")
        print(file[0])

        lines = file[1].strip().split('.')
        ctr = 0
        max_length = len(lines)
        if max_length > 1:
            f = open("..\political_corpus\\" + file[0].strip(), 'a', encoding='utf8')
            for per_line in lines:
                if ctr == 0 and len(per_line.strip().split('-')) > 2:
                    f.write(per_line.strip().split('-')[1].strip() + ".\n")
                elif "Baca juga:" not in per_line.strip() and \
                        "[Gambas:Instagram]" not in per_line.strip() and \
                        "detikFlash" not in per_line.strip() and \
                        ctr < (max_length - 1):
                    f.write(per_line.strip() + ".\n")
                ctr += 1
            f.close()
        # break


if __name__ == '__main__':
    conn = create_connection()
    data = get_political_news(conn)
    create_per_file(data)
