package com.example.dragonassist.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * A fixed palette rather than Material You's wallpaper-derived colours.
 *
 * Dynamic colour is pleasant on a personal device but wrong for this app: the UI would
 * look different on every phone and change when someone sets a new wallpaper, which is
 * not what you want ten minutes before a demo.
 *
 * Ember and warm neutrals, with a slate blue for the user's own speech so the two chat
 * bubbles stay distinguishable in both schemes.
 */

// Light
val EmberPrimaryLight = Color(0xFF8F3D19)
val EmberOnPrimaryLight = Color(0xFFFFFFFF)
val EmberContainerLight = Color(0xFFFFDBCB)
val EmberOnContainerLight = Color(0xFF351000)

val SlateSecondaryLight = Color(0xFF3E5F78)
val SlateOnSecondaryLight = Color(0xFFFFFFFF)
val SlateContainerLight = Color(0xFFC6E7FF)
val SlateOnContainerLight = Color(0xFF001E2F)

val GoldTertiaryLight = Color(0xFF6C5E10)
val GoldOnTertiaryLight = Color(0xFFFFFFFF)
val GoldContainerLight = Color(0xFFF6E28B)
val GoldOnContainerLight = Color(0xFF211B00)

val SurfaceLight = Color(0xFFFFF8F6)
val OnSurfaceLight = Color(0xFF221A16)
val SurfaceVariantLight = Color(0xFFF4DED5)
val OnSurfaceVariantLight = Color(0xFF52443D)
val OutlineLight = Color(0xFF85736C)

// Dark
val EmberPrimaryDark = Color(0xFFFFB68F)
val EmberOnPrimaryDark = Color(0xFF542100)
val EmberContainerDark = Color(0xFF723200)
val EmberOnContainerDark = Color(0xFFFFDBCB)

val SlateSecondaryDark = Color(0xFFA6C8E3)
val SlateOnSecondaryDark = Color(0xFF0A3449)
val SlateContainerDark = Color(0xFF274B62)
val SlateOnContainerDark = Color(0xFFC6E7FF)

val GoldTertiaryDark = Color(0xFFD9C66F)
val GoldOnTertiaryDark = Color(0xFF3A3000)
val GoldContainerDark = Color(0xFF524600)
val GoldOnContainerDark = Color(0xFFF6E28B)

// Deliberately not pure black: OLED black makes the elevated cards read as floating
// rectangles, and a slight warm tint keeps them attached to the surface.
val SurfaceDark = Color(0xFF1A120E)
val OnSurfaceDark = Color(0xFFF1DFD8)
val SurfaceVariantDark = Color(0xFF52443D)
val OnSurfaceVariantDark = Color(0xFFD7C2B9)
val OutlineDark = Color(0xFF9F8D85)

val ErrorLight = Color(0xFFBA1A1A)
val ErrorContainerLight = Color(0xFFFFDAD6)
val OnErrorContainerLight = Color(0xFF410002)
val ErrorDark = Color(0xFFFFB4AB)
val ErrorContainerDark = Color(0xFF93000A)
val OnErrorContainerDark = Color(0xFFFFDAD6)
