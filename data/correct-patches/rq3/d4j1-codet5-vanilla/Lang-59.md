# Repilot Patch

```

                str.getChars(0, width, buffer, size);
```

# Developer Patch

```
                str.getChars(0, width, buffer, size);
```

# Context

```
--- bug/Lang-59/src/java/org/apache/commons/lang/text/StrBuilder.java

+++ fix/Lang-59/src/java/org/apache/commons/lang/text/StrBuilder.java

@@ -881,7 +881,8 @@

             String str = (obj == null ? getNullText() : obj.toString());
             int strLen = str.length();
             if (strLen >= width) {
-                str.getChars(0, strLen, buffer, size);
+
+                str.getChars(0, width, buffer, size);
             } else {
                 int padLen = width - strLen;
                 str.getChars(0, strLen, buffer, size);
```

# Note

