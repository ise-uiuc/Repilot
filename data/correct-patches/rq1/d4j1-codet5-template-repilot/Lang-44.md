# Repilot Patch

```

        if (val.length() == 1) {
            return createInteger(val);
        }
```

# Developer Patch

```
        if (val.length() == 1 && !Character.isDigit(val.charAt(0))) {
            throw new NumberFormatException(val + " is not a valid number.");
        }
```

# Context

```
--- bug/Lang-44/src/java/org/apache/commons/lang/NumberUtils.java

+++ fix/Lang-44/src/java/org/apache/commons/lang/NumberUtils.java

@@ -142,6 +142,10 @@

         if (val.length() == 0) {
             throw new NumberFormatException("\"\" is not a valid number.");
         }
+
+        if (val.length() == 1) {
+            return createInteger(val);
+        }
         if (val.startsWith("--")) {
             // this is protection for poorness in java.lang.BigDecimal.
             // it accepts this as a legal value, but it does not appear
```

# Note

