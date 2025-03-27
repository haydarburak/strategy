import datetime

# Get current date and time
now = datetime.datetime.now()

# Format timestamp
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

# Log the timestamp
log_message = f"Script ran at: {timestamp}\n"

# Print to console (GitHub Actions log)
print(log_message)

# Save to a file
with open("log.txt", "a") as file:
    file.write(log_message)
