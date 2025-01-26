"""
# Data Flow for Inventory Selection

## 1. Input Data
# - st.session_state.original_inventory_data: Original inventory dataset
#   Structure:
#   - product_id: Product ID
#   - product_name: Product name
#   - category: Category
#   - brand: Brand
#   - stock_quantity: Stock quantity
#   - retail_price: Retail price
#   - cost_price: Cost price
#   - profit_margin: Profit margin in percent

## 2. Filter and Sort Products
# Input: st.session_state.original_inventory_data
# Process:
#   1. Create a copy of data: df = st.session_state.original_inventory_data.copy()
#   2. Apply selected filters (can be combined):
#      - show_high_stock: Top N products by stock_quantity
#      - show_high_margin: Top N products by profit_margin
#      - show_high_price: Top N products by retail_price
#      N = num_products (user-selected, 5-50)
#   3. Combine filtered products using sets to avoid duplicates
# Output: filtered_df - DataFrame containing products matching any selected filter

## 3. Product Selection
# Input: filtered_df from step 2
# Process:
#   1. User selects product from dropdown (includes empty option)
#   2. When product is selected:
#      selected_product_info = filtered_df[filtered_df["product_name"] == selected_product].iloc[0]
# Output: st.session_state.selected_product_info
#   Contains all product information:
#   - product_id
#   - product_name
#   - category
#   - brand
#   - stock_quantity
#   - retail_price
#   - cost_price
#   - profit_margin

## 4. Navigation
# Input: st.session_state.selected_product_info
# Process: 
#   1. Check if product is selected
#   2. If selected, enable navigation to next section
# Output: 
#   - st.session_state.go_to_email = True (only if product is selected)
#   - Error message if no product selected

## Session State Variables:
# - original_inventory_data: Original dataset (input)
# - selected_product_info: Selected product details (used in Email Design section)
# - go_to_email: Navigation flag for next section
""" 