# Repilot Patch

```

            } else if (contains(value, index + 1, 3, "IER")) {
```

# Developer Patch

```
            } else if (contains(value, index + 1, 3, "IER")) {
```

# Context

```
--- bug/Codec-3/src/java/org/apache/commons/codec/language/DoubleMetaphone.java

+++ fix/Codec-3/src/java/org/apache/commons/codec/language/DoubleMetaphone.java

@@ -452,7 +452,8 @@

             if ((contains(value, 0 ,4, "VAN ", "VON ") || contains(value, 0, 3, "SCH")) || contains(value, index + 1, 2, "ET")) {
                 //-- obvious germanic --//
                 result.append('K');
-            } else if (contains(value, index + 1, 4, "IER")) {
+
+            } else if (contains(value, index + 1, 3, "IER")) {
                 result.append('J');
             } else {
                 result.append('J', 'K');
```

# Note

