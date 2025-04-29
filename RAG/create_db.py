import sqlite3

with open("Chinook_Sqlite.sql", "r", encoding="utf-8") as f:
    sql_script = f.read()

conn = sqlite3.connect("Chinook.db")
cursor = conn.cursor()
cursor.executescript(sql_script)
conn.commit()
conn.close()

print("Chinook.db created successfully.")
