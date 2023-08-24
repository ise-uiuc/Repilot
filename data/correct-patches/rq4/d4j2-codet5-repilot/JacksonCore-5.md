# Repilot Patch

```

            char c = str.charAt(i);
```

# Developer Patch

```
            char c = str.charAt(i);
```

# Context

```
--- bug/JacksonCore-5/src/main/java/com/fasterxml/jackson/core/JsonPointer.java

+++ fix/JacksonCore-5/src/main/java/com/fasterxml/jackson/core/JsonPointer.java

@@ -190,7 +190,8 @@

             return -1;
         }
         for (int i = 0; i < len; ++i) {
-            char c = str.charAt(i++);
+
+            char c = str.charAt(i);
             if (c > '9' || c < '0') {
                 return -1;
             }
```

# Note

