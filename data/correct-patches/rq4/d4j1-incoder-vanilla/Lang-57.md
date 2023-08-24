# Repilot Patch

```
        return Arrays.asList(Locale.getAvailableLocales()).contains(locale);
```

# Developer Patch

```
        return availableLocaleList().contains(locale);
```

# Context

```
--- bug/Lang-57/src/java/org/apache/commons/lang/LocaleUtils.java

+++ fix/Lang-57/src/java/org/apache/commons/lang/LocaleUtils.java

@@ -220,7 +220,7 @@

      * @return true if the locale is a known locale
      */
     public static boolean isAvailableLocale(Locale locale) {
-        return cAvailableLocaleSet.contains(locale);
+        return Arrays.asList(Locale.getAvailableLocales()).contains(locale);
     }
 
     //-----------------------------------------------------------------------
```

# Note

