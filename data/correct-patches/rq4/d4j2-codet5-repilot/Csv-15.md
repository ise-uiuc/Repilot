# Repilot Patch

```

                if (c <= COMMENT) {
```

# Developer Patch

```
                if (c <= COMMENT) {
```

# Context

```
--- bug/Csv-15/src/main/java/org/apache/commons/csv/CSVFormat.java

+++ fix/Csv-15/src/main/java/org/apache/commons/csv/CSVFormat.java

@@ -1186,9 +1186,8 @@

             } else {
                 char c = value.charAt(pos);
 
-                if (newRecord && (c < 0x20 || c > 0x21 && c < 0x23 || c > 0x2B && c < 0x2D || c > 0x7E)) {
-                    quote = true;
-                } else if (c <= COMMENT) {
+
+                if (c <= COMMENT) {
                     // Some other chars at the start of a value caused the parser to fail, so for now
                     // encapsulate if we start in anything less than '#'. We are being conservative
                     // by including the default comment char too.
```

# Note

