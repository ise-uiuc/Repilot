# Repilot Patch

```
return newString(bytes, Charsets.ISO_8859_1);
```

# Developer Patch

```
        return newString(bytes, Charsets.ISO_8859_1);
```

# Context

```
--- bug/Codec-17/src/main/java/org/apache/commons/codec/binary/StringUtils.java

+++ fix/Codec-17/src/main/java/org/apache/commons/codec/binary/StringUtils.java

@@ -336,7 +336,7 @@

      * @since As of 1.7, throws {@link NullPointerException} instead of UnsupportedEncodingException
      */
     public static String newStringIso8859_1(final byte[] bytes) {
-        return new String(bytes, Charsets.ISO_8859_1);
+return newString(bytes, Charsets.ISO_8859_1);
     }
 
     /**
```

# Note

