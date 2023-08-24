# Repilot Patch

```
                if (input.length <= end) {
                    return false;
                }
```

# Developer Patch

```
                if (end >= input.length)
                    return false;
```

# Context

```
--- bug/Jsoup-90/src/main/java/org/jsoup/helper/HttpConnection.java

+++ fix/Jsoup-90/src/main/java/org/jsoup/helper/HttpConnection.java

@@ -421,6 +421,9 @@

                     return false;
                 }
 
+                if (input.length <= end) {
+                    return false;
+                }
 
                 while (i < end) {
                     i++;
```

# Note

