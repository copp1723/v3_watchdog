# Watchdog AI Theme System

This directory contains the CSS styling for the Watchdog AI UI, with a focus on a themeable design system using CSS variables.

## Overview

The theme system is designed to support both light and dark modes through a single set of CSS variables defined in `theme.css`. All UI components should reference these variables instead of using hard-coded values to ensure consistent styling and easy theme switching.

## Files

- **theme.css** - Defines all theme variables for both light and dark modes
- **main.css** - Contains global styling and common components
- **insight_card.css** - Styles specific to insight cards

## How to Use

### Applying Theme

The theme is applied by adding the `.light-theme` class to the `<body>` element. The header component handles toggling this class through the theme toggle button.

```javascript
document.body.className = 'light-theme'; // For light theme
document.body.className = ''; // For dark theme
```

### Using CSS Variables

Always use CSS variables for all styling properties:

```css
/* ❌ Don't use hard-coded values */
.my-component {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #dcdcdc;
}

/* ✅ Use CSS variables instead */
.my-component {
    background-color: var(--bg-card);
    color: var(--fg-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: var(--spacing-md);
}
```

## Variable Reference

### Colors

- `--bg-primary` - Primary background color
- `--bg-secondary` - Secondary background color
- `--bg-card` - Card background color
- `--bg-panel` - Panel background color
- `--bg-input` - Input field background color
- `--bg-page` - Page background color

- `--fg-primary` - Primary text color
- `--fg-secondary` - Secondary text color
- `--fg-muted` - Muted text color
- `--fg-placeholder` - Placeholder text color

- `--accent-primary` - Primary accent color
- `--accent-warning` - Warning accent color
- `--accent-success` - Success accent color
- `--accent-high` - High-priority accent (usually red)
- `--accent-medium` - Medium-priority accent (usually yellow/orange)
- `--accent-info` - Info accent (usually blue)

### Status Colors

- `--status-error` - Error status color
- `--status-warning` - Warning status color 
- `--status-info` - Info status color
- `--status-success` - Success status color

### Spacing

- `--spacing-xs` - Extra small spacing (0.25rem)
- `--spacing-sm` - Small spacing (0.5rem)
- `--spacing-md` - Medium spacing (1rem)
- `--spacing-lg` - Large spacing (1.5rem)
- `--spacing-xl` - Extra large spacing (2rem)

### Typography

- `--font-sans` - Sans-serif font stack
- `--font-size-xs` - Extra small font size (0.75rem)
- `--font-size-sm` - Small font size (0.875rem)
- `--font-size-md` - Medium font size (1rem)
- `--font-size-lg` - Large font size (1.25rem)
- `--font-size-xl` - Extra large font size (1.5rem)

### Borders & Shadows

- `--border-color` - Border color
- `--border-radius` - Border radius
- `--shadow-card` - Card shadow
- `--shadow-panel` - Panel shadow

### Button Gradients

- `--btn-gradient` - Button gradient
- `--btn-gradient-hover` - Button hover gradient

## Adding New Variables

When adding new CSS variables, always add them to both themes in `theme.css`:

1. Add the variable to the `:root` selector (dark theme)
2. Add the corresponding variable to the `.light-theme` selector
3. Document the variable purpose with a comment

## Best Practices

1. Never use hard-coded colors, spacing, or typography values
2. Keep related styles in appropriate CSS files
3. Use semantic variable names (e.g., `--accent-primary` instead of `--purple`)
4. Test in both light and dark themes
5. Ensure adequate contrast in both themes
6. Use responsive design with media queries

## Streamlit-specific Styling

Some Streamlit components require special handling. Use the `styles.py` utility functions for Streamlit-specific styling.