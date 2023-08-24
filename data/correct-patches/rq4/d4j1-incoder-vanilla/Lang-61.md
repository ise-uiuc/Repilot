# Repilot Patch

```
        int len = size - strLen + 1;
```

# Developer Patch

```
        int len = size - strLen + 1;
```

# Context

```
--- bug/Lang-61/src/java/org/apache/commons/lang/text/StrBuilder.java

+++ fix/Lang-61/src/java/org/apache/commons/lang/text/StrBuilder.java

@@ -1773,7 +1773,7 @@

             return -1;
         }
         char[] thisBuf = buffer;
-        int len = thisBuf.length - strLen;
+        int len = size - strLen + 1;
         outer:
         for (int i = startIndex; i < len; i++) {
             for (int j = 0; j < strLen; j++) {
```

# Note

