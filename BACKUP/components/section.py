def create_section(title, content, emoji="", extra_classes=""):
    """
    Creates a styled section with title and content.
    
    Args:
        title (str): The section title
        content (str): The HTML content of the section
        emoji (str): Optional emoji to display before the title
        extra_classes (str): Optional additional CSS classes
    
    Returns:
        str: HTML string for the section
    """
    return f"""
        <div class="section-container {extra_classes}">
            <h3 class="section-header">{emoji} {title}</h3>
            {content}
        </div>
    """ 