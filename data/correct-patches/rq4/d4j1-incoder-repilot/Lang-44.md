# Repilot Patch

```
        if (val.length() < 2) { //short circuit for "0" and "-0"
            return createLong(val);
        }
        //Check for leading negative sign - JDK 1.4 has known issues with BigDecimal.
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
+        if (val.length() < 2) { //short circuit for "0" and "-0"
+            return createLong(val);
+        }
+        //Check for leading negative sign - JDK 1.4 has known issues with BigDecimal.
         if (val.startsWith("--")) {
             // this is protection for poorness in java.lang.BigDecimal.
             // it accepts this as a legal value, but it does not appear
```

# Note

`createLong` subsumes the check, and `< 2` is equivalent to `== 1` in this context.