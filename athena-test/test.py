# test.py

def main():
    print("Welcome to the simple calculator!")

    # Input two numbers
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))

    # Calculate the sum
    result = num1 + num2

    # Display the result
    print(f"The sum of {num1} and {num2} is {result}")

# Ensure that the main() function is called only when this script is executed directly
if __name__ == "__main__":
    main()
