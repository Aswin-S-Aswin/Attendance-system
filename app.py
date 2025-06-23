import os

def main_menu():
    print("\n==== Office Attendance System ====")
    print("1. Register New Employee")
    print("2. Train Recognition Model")
    print("3. Mark Attendance")
    print("4. Exit")
    return input("Select option: ")

if __name__ == "__main__":
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("trained_model", exist_ok=True)
    
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp"])
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            from register_employee import register_employee
            emp_id = input("Employee ID: ")
            name = input("Full Name: ")
            register_employee(emp_id, name)
            
        elif choice == '2':
            from train_model import train_model
            train_model()
            
        elif choice == '3':
            from mark_attendance import mark_attendance
            mark_attendance()
            
        elif choice == '4':
            print("Exiting system...")
            break
            
        else:
            print("Invalid choice! Please try again.")
