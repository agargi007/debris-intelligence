# Design System Specification: Deep Sea Intelligence

## 1. Overview & Creative North Star
**Creative North Star: "The Submerged Observer"**
This design system moves away from the "flat dashboard" trope, instead adopting the perspective of a high-tech submersible navigating the midnight zone. The aesthetic is defined by **Luminous Depth**—the contrast between the crushing darkness of the deep ocean (`surface`) and the hyper-precise, glowing data overlays (`primary`). 

We break the standard grid by using **intentional asymmetry** and **overlapping glass layers**. Content should feel like it is floating in a fluid environment, using depth and light—rather than rigid lines—to guide the eye. Every interface element must feel like an optical instrument: precise, refined, and vital.

---

## 2. Colors & Atmospheric Depth

### The Palette
The core of this system is a range of deep navies and electric cyans, optimized for a high-contrast dark mode.

*   **Backgrounds:** `surface` (#070d1f) to `surface_container_low` (#0c1326).
*   **Primary Accents:** `primary` (#39b8fd) and `secondary` (#2db7f2).
*   **Data Visualization (Debris Types):**
    *   **Plastic:** `tertiary_fixed` (#65fde6) - A bright, synthetic mint.
    *   **Metal:** `primary_fixed` (#34b5fa) - A structural, cool blue.
    *   **Rubber:** `inverse_on_surface` (#4f5469) - A matte, dark neutral.
    *   **Glass:** `on_surface_variant` (#a5aac2) - Translucent grey.
    *   **Fabric:** `error_dim` (#d7383b) - A warning-level contrast for organic-synthetic entanglement.

### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders to define sections. Layouts must be partitioned using:
1.  **Tonal Shifts:** Place a `surface_container_high` module against a `surface` background.
2.  **Negative Space:** Utilize the `12` (3rem) and `16` (4rem) spacing tokens to separate concepts.
3.  **Luminous Glow:** Use a subtle outer glow (0px 0px 15px) using `primary` at 10% opacity to define a container's presence.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers of water and glass.
*   **Base:** `surface` (The deep ocean).
*   **Mid-Ground:** `surface_container` (The navigation HUD).
*   **Fore-Ground:** `surface_container_highest` (Active modal or critical alert).
*   **Nesting:** When placing a card inside a section, the card should be *lighter* than the section background to create "upward" buoyancy.

---

## 3. Typography: The Editorial Precision
We use **Inter** exclusively, but we treat it with editorial weight.

*   **Display (Lg/Md):** Used for "Big Data" hero numbers (e.g., total debris detected). Letter-spacing should be set to `-0.02em` for a compact, technical feel.
*   **Headline (Sm/Md):** Used for section titles. Pair these with `label-sm` in all-caps for a "military-grade" HUD aesthetic.
*   **Body (Md):** The workhorse. Use `on_surface_variant` (#a5aac2) for long-form reading to reduce eye strain against the dark background.
*   **Labels:** Always use `label-md` or `label-sm` for metadata. This conveys the precision of an AI sensor readout.

---

## 4. Elevation & Depth: The Glass Principle

### The Layering Principle
Depth is achieved through **Tonal Layering**. Avoid traditional drop shadows.
*   **Level 1 (Flat):** `surface`
*   **Level 2 (Floating):** `surface_container_low` + Backdrop Blur (20px).
*   **Level 3 (Interactive):** `surface_container_high` + `primary` Ghost Border.

### Glassmorphism & Ghost Borders
To achieve the premium "submersible HUD" look, all floating panels must use:
*   **Background:** `surface_container` at 70% opacity.
*   **Backdrop Filter:** `blur(24px)`.
*   **The Ghost Border:** Instead of a solid line, use `outline_variant` at 20% opacity. This creates a "specular highlight" on the edge of the glass rather than a physical boundary.

### Ambient Shadows
If a floating element requires a shadow, it must be an "Oceanic Shadow": 
*   **Color:** `surface_container_lowest` (#000000) at 40% opacity.
*   **Spread:** Large blur (40px-60px), 0px offset. It should feel like a silhouette in deep water, not a shadow on a wall.

---

## 5. Components

### The Primary Action (The "Sonar" Button)
*   **Base:** `primary` (#39b8fd) gradient transitioning to `primary_container`.
*   **Corner:** `xl` (1.5rem) for a smooth, organic feel.
*   **State:** On hover, increase `surface_tint` glow. On click, scale to 98%.

### Glass Cards
*   **Structure:** No dividers. Use `surface_container_lowest` for the header area and `surface_container` for the body to create a natural visual break.
*   **Corners:** `xl` (1.5rem).

### Data Chips
*   **Style:** `surface_variant` background with `on_surface` text.
*   **Feature:** Add a 4px circular dot of the data-viz color (e.g., Plastic Mint) to the left of the text for instant classification.

### AI Detection Inputs
*   **Style:** Ghost Borders only. The input field should be `surface_container_low`. 
*   **Focus State:** The border transitions from 20% opacity to 100% `primary` with a subtle 4px outer glow.

### New Component: The "Depth Gauge" (Progress/Loading)
Instead of a circular spinner, use a vertical "Depth Gauge." A thin bar (2px) using `outline_variant`, with a `primary` glow filling it from top to bottom, mimicking a probe descending into the sea.

---

## 6. Do’s and Don’ts

### Do:
*   **Use Asymmetry:** Place a large `display-lg` metric off-center to create a modern, editorial feel.
*   **Embrace "Air":** Use the `20` (5rem) spacing token between major modules to allow the "ocean" (background) to breathe.
*   **Layer Glass:** Overlap a small glass panel (e.g., a filter) over a larger one to create depth.

### Don’t:
*   **No Pure White:** Never use #FFFFFF. Use `on_surface` (#dfe4fe). Pure white breaks the "submerged" immersion.
*   **No 1px Dividers:** Never use a line to separate list items. Use a 4px `0.25rem` gap and a slight background shift on hover.
*   **No Sharp Corners:** Avoid `none` or `sm` roundedness. The ocean is fluid; the UI should be too. Stick to `xl` and `2xl`.